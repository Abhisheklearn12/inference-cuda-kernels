#pragma once
// Single-query attention decode ("one new token attends to the KV cache"),
// fp32, batch 1:
//   for each head h:  out[h] = softmax(q[h] . K[h]^T / sqrt(D)) @ V[h]
// Layouts: q, out are [H, D]; K, V are [H, L, D], all row-major.
//
// This is the attention pattern of every autoregressive decode step. It is
// memory bound: the whole KV cache (2*H*L*D floats) must be streamed once
// per token, so the roofline is again DRAM bandwidth.

#include "../common.cuh"

// ---------------------------------------------------------------------------
// Kernel 1: one block per head, full score vector materialized in shared
// memory, three phases (scores, softmax, weighted sum of V).
// Correct and simple, but shared memory must hold L floats, which caps the
// sequence length at about 12k (48 KB default dynamic smem limit), and the
// q.K dot products are computed one-thread-per-position with strided,
// poorly coalesced K reads.
// Dynamic smem: (L + D + 32) floats.
// ---------------------------------------------------------------------------
__global__ void attention_k1_naive(const float* __restrict__ q,
                                   const float* __restrict__ K,
                                   const float* __restrict__ V,
                                   float* __restrict__ out, int H, int L,
                                   int D, float scale) {
  extern __shared__ float smem[];
  float* sc = smem;          // L scores
  float* qs = smem + L;      // D query values
  float* red = qs + D;       // 32 reduction slots
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const float* Kh = K + (size_t)head * L * D;
  const float* Vh = V + (size_t)head * L * D;

  for (int d = tid; d < D; d += blockDim.x) qs[d] = q[(size_t)head * D + d];
  __syncthreads();

  // Phase 1: scores. Thread t handles positions t, t+blockDim, ...
  for (int pos = tid; pos < L; pos += blockDim.x) {
    float dot = 0.0f;
    const float* kp = Kh + (size_t)pos * D;
    for (int d = 0; d < D; d++) dot += qs[d] * kp[d];
    sc[pos] = dot * scale;
  }

  // Phase 2: safe softmax over the scores (in place).
  float m = -FLT_MAX;
  for (int pos = tid; pos < L; pos += blockDim.x) m = fmaxf(m, sc[pos]);
  m = block_reduce_max(m, red);  // internal barriers also publish sc[]
  float s = 0.0f;
  for (int pos = tid; pos < L; pos += blockDim.x) {
    float p = expf(sc[pos] - m);
    sc[pos] = p;
    s += p;
  }
  s = block_reduce_sum(s, red);
  const float inv = 1.0f / s;

  // Phase 3: out[d] = sum_pos p[pos] * V[pos, d]. Thread d reads a column of
  // V; consecutive threads read consecutive addresses, so this is coalesced.
  for (int d = tid; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int pos = 0; pos < L; pos++) acc += sc[pos] * Vh[(size_t)pos * D + d];
    out[(size_t)head * D + d] = acc * inv;
  }
}

// ---------------------------------------------------------------------------
// Kernel 2: flash-decoding style. One block per head, KV cache processed in
// tiles of ATTN_TILE positions with an online softmax: running (max m,
// sum s) plus a rescaled output accumulator, so the full score vector is
// never materialized and L is unbounded.
// Score dot products are computed one-warp-per-position with float4 loads
// and a shuffle reduction, so K reads are wide and coalesced within the warp.
// Dynamic smem: (ATTN_TILE + 2*D + 32) floats.
// ---------------------------------------------------------------------------
constexpr int ATTN_TILE = 128;

__global__ void attention_k2_flash(const float* __restrict__ q,
                                   const float* __restrict__ K,
                                   const float* __restrict__ V,
                                   float* __restrict__ out, int H, int L,
                                   int D, float scale) {
  extern __shared__ float smem[];
  float* sc = smem;               // ATTN_TILE tile scores
  float* qs = sc + ATTN_TILE;     // D query values
  float* acc = qs + D;            // D output accumulators
  float* red = acc + D;           // 32 reduction slots
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  const float* Kh = K + (size_t)head * L * D;
  const float* Vh = V + (size_t)head * L * D;

  for (int d = tid; d < D; d += blockDim.x) {
    qs[d] = q[(size_t)head * D + d];
    acc[d] = 0.0f;
  }
  __syncthreads();

  // Every thread carries identical copies of the running statistics; they
  // stay identical because both are only updated from block-wide reductions.
  float m_run = -FLT_MAX;
  float s_run = 0.0f;

  for (int start = 0; start < L; start += ATTN_TILE) {
    const int tl = min(ATTN_TILE, L - start);

    // Tile scores: warp w handles positions w, w+nwarps, ... of the tile.
    for (int t = wid; t < tl; t += nwarps) {
      const float* kp = Kh + (size_t)(start + t) * D;
      float dot = 0.0f;
      if ((D & 3) == 0) {
        const float4* kp4 = reinterpret_cast<const float4*>(kp);
        const float4* qs4 = reinterpret_cast<const float4*>(qs);
        for (int i = lane; i < (D >> 2); i += 32) {
          float4 kv = kp4[i];
          float4 qv = qs4[i];
          dot += kv.x * qv.x + kv.y * qv.y + kv.z * qv.z + kv.w * qv.w;
        }
      } else {
        for (int d = lane; d < D; d += 32) dot += qs[d] * kp[d];
      }
      dot = warp_reduce_sum(dot);
      if (lane == 0) sc[t] = dot * scale;
    }
    __syncthreads();

    // Online softmax update for this tile.
    float tm = -FLT_MAX;
    for (int t = tid; t < tl; t += blockDim.x) tm = fmaxf(tm, sc[t]);
    tm = block_reduce_max(tm, red);
    const float m_new = fmaxf(m_run, tm);
    const float corr = expf(m_run - m_new);  // 0 on the first tile

    float ts = 0.0f;
    for (int t = tid; t < tl; t += blockDim.x) {
      float p = expf(sc[t] - m_new);
      sc[t] = p;
      ts += p;
    }
    ts = block_reduce_sum(ts, red);  // barriers also publish the new sc[]
    s_run = s_run * corr + ts;
    m_run = m_new;

    // Rescale the accumulator and add this tile's p . V contribution.
    // Thread d walks a column of V: coalesced across threads.
    for (int d = tid; d < D; d += blockDim.x) {
      float a = acc[d] * corr;
      for (int t = 0; t < tl; t++)
        a += sc[t] * Vh[(size_t)(start + t) * D + d];
      acc[d] = a;
    }
    __syncthreads();  // sc[] must not be overwritten before all reads finish
  }

  const float inv = 1.0f / s_run;
  for (int d = tid; d < D; d += blockDim.x)
    out[(size_t)head * D + d] = acc[d] * inv;
}

// ---------------------------------------------------------------------------
// Kernel 3: flash-decoding with split-KV (Dao et al. 2023). Kernel 2 launches
// only H blocks, which cannot fill 28 SMs when H is small (H=32 leaves most
// of the GPU idle). Here the sequence is additionally split S ways: block
// (h, s) runs the kernel-2 loop over its chunk of the KV cache and writes an
// unnormalized partial (max m, sum s, accumulator acc[D]) to global memory.
// A small second kernel merges the S partials per head with the same online
// softmax rescaling rule. Grid size becomes H*S blocks.
// Partial layouts: pm, ps are [H, S]; pacc is [H, S, D].
// Dynamic smem: (ATTN_TILE + 2*D + 32) floats.
// ---------------------------------------------------------------------------
__global__ void attention_k3_split(const float* __restrict__ q,
                                   const float* __restrict__ K,
                                   const float* __restrict__ V,
                                   float* __restrict__ pm,
                                   float* __restrict__ ps,
                                   float* __restrict__ pacc, int H, int L,
                                   int D, float scale, int S) {
  extern __shared__ float smem[];
  float* sc = smem;
  float* qs = sc + ATTN_TILE;
  float* acc = qs + D;
  float* red = acc + D;
  const int head = blockIdx.x;
  const int split = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  const size_t pidx = (size_t)head * S + split;

  const int chunk = (L + S - 1) / S;
  const int begin = split * chunk;
  const int end = min(L, begin + chunk);
  if (begin >= end) {  // empty chunk: neutral element of the merge
    for (int d = tid; d < D; d += blockDim.x) pacc[pidx * D + d] = 0.0f;
    if (tid == 0) {
      pm[pidx] = -FLT_MAX;
      ps[pidx] = 0.0f;
    }
    return;
  }

  const float* Kh = K + (size_t)head * L * D;
  const float* Vh = V + (size_t)head * L * D;
  for (int d = tid; d < D; d += blockDim.x) {
    qs[d] = q[(size_t)head * D + d];
    acc[d] = 0.0f;
  }
  __syncthreads();

  float m_run = -FLT_MAX;
  float s_run = 0.0f;

  for (int start = begin; start < end; start += ATTN_TILE) {
    const int tl = min(ATTN_TILE, end - start);

    for (int t = wid; t < tl; t += nwarps) {
      const float* kp = Kh + (size_t)(start + t) * D;
      float dot = 0.0f;
      if ((D & 3) == 0) {
        const float4* kp4 = reinterpret_cast<const float4*>(kp);
        const float4* qs4 = reinterpret_cast<const float4*>(qs);
        for (int i = lane; i < (D >> 2); i += 32) {
          float4 kv = kp4[i];
          float4 qv = qs4[i];
          dot += kv.x * qv.x + kv.y * qv.y + kv.z * qv.z + kv.w * qv.w;
        }
      } else {
        for (int d = lane; d < D; d += 32) dot += qs[d] * kp[d];
      }
      dot = warp_reduce_sum(dot);
      if (lane == 0) sc[t] = dot * scale;
    }
    __syncthreads();

    float tm = -FLT_MAX;
    for (int t = tid; t < tl; t += blockDim.x) tm = fmaxf(tm, sc[t]);
    tm = block_reduce_max(tm, red);
    const float m_new = fmaxf(m_run, tm);
    const float corr = expf(m_run - m_new);

    float ts = 0.0f;
    for (int t = tid; t < tl; t += blockDim.x) {
      float p = expf(sc[t] - m_new);
      sc[t] = p;
      ts += p;
    }
    ts = block_reduce_sum(ts, red);
    s_run = s_run * corr + ts;
    m_run = m_new;

    for (int d = tid; d < D; d += blockDim.x) {
      float a = acc[d] * corr;
      for (int t = 0; t < tl; t++)
        a += sc[t] * Vh[(size_t)(start + t) * D + d];
      acc[d] = a;
    }
    __syncthreads();
  }

  for (int d = tid; d < D; d += blockDim.x) pacc[pidx * D + d] = acc[d];
  if (tid == 0) {
    pm[pidx] = m_run;
    ps[pidx] = s_run;
  }
}

// Merge kernel: one block per head combines the S partials.
__global__ void attention_k3_reduce(const float* __restrict__ pm,
                                    const float* __restrict__ ps,
                                    const float* __restrict__ pacc,
                                    float* __restrict__ out, int H, int D,
                                    int S) {
  const int head = blockIdx.x;
  const size_t base = (size_t)head * S;

  // Every thread redundantly computes the merged (max, sum); S is tiny.
  float m = -FLT_MAX;
  for (int i = 0; i < S; i++) m = fmaxf(m, pm[base + i]);
  float s = 0.0f;
  for (int i = 0; i < S; i++) s += ps[base + i] * expf(pm[base + i] - m);
  const float inv = 1.0f / s;

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float a = 0.0f;
    for (int i = 0; i < S; i++)
      a += pacc[(base + i) * D + d] * expf(pm[base + i] - m);
    out[(size_t)head * D + d] = a * inv;
  }
}

#pragma once
// Row-wise numerically safe softmax on a [M, N] matrix, fp32.
//   y[i,j] = exp(x[i,j] - max_j x[i,j]) / sum_j exp(x[i,j] - max_j x[i,j])
//
// Used on attention scores and on the final logits during inference. Also
// memory bound: a handful of exp() per element never saturates the SFU
// before DRAM saturates. The interesting axis is how many times each row
// is read from memory: the safe formulation naively needs 3 passes (max,
// sum, normalize); the online formulation fuses max and sum into one pass.

#include "../common.cuh"

// ---------------------------------------------------------------------------
// Kernel 1: naive, one thread per row, 3 sequential passes.
// Zero parallelism inside a row and uncoalesced access across the warp,
// exactly like gemv_k1. With few rows the GPU is almost idle.
// ---------------------------------------------------------------------------
__global__ void softmax_k1_naive(const float* x, float* y, int M, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;
  const float* xr = x + (size_t)row * N;
  float* yr = y + (size_t)row * N;
  float m = -FLT_MAX;
  for (int j = 0; j < N; j++) m = fmaxf(m, xr[j]);
  float s = 0.0f;
  for (int j = 0; j < N; j++) s += expf(xr[j] - m);
  float inv = 1.0f / s;
  for (int j = 0; j < N; j++) yr[j] = expf(xr[j] - m) * inv;
}

// ---------------------------------------------------------------------------
// Kernel 2: one warp per row, shuffle reductions. Coalesced, still 3 passes.
// ---------------------------------------------------------------------------
__global__ void softmax_k2_warp(const float* __restrict__ x,
                                float* __restrict__ y, int M, int N) {
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= M) return;
  const float* xr = x + (size_t)row * N;
  float* yr = y + (size_t)row * N;

  float m = -FLT_MAX;
  for (int j = lane; j < N; j += 32) m = fmaxf(m, xr[j]);
  m = warp_reduce_max(m);

  float s = 0.0f;
  for (int j = lane; j < N; j += 32) s += expf(xr[j] - m);
  s = warp_reduce_sum(s);

  const float inv = 1.0f / s;
  for (int j = lane; j < N; j += 32) yr[j] = expf(xr[j] - m) * inv;
}

// ---------------------------------------------------------------------------
// Kernel 3: one block (256 threads) per row, float4 loads, block reductions.
// A whole block per row keeps the SMs busy even for small M (e.g. logits
// with batch 1) and vector loads maximize transaction width. Still 3 passes.
// ---------------------------------------------------------------------------
__global__ void softmax_k3_block_vec4(const float* __restrict__ x,
                                      float* __restrict__ y, int M, int N) {
  __shared__ float red[32];
  const int row = blockIdx.x;
  if (row >= M) return;
  const float* xr = x + (size_t)row * N;
  float* yr = y + (size_t)row * N;
  const bool vec = (N & 3) == 0;
  const int N4 = N >> 2;

  float m = -FLT_MAX;
  if (vec) {
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    for (int i = threadIdx.x; i < N4; i += blockDim.x) {
      float4 v = xr4[i];
      m = fmaxf(m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
  } else {
    for (int j = threadIdx.x; j < N; j += blockDim.x) m = fmaxf(m, xr[j]);
  }
  m = block_reduce_max(m, red);

  float s = 0.0f;
  if (vec) {
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    for (int i = threadIdx.x; i < N4; i += blockDim.x) {
      float4 v = xr4[i];
      s += expf(v.x - m) + expf(v.y - m) + expf(v.z - m) + expf(v.w - m);
    }
  } else {
    for (int j = threadIdx.x; j < N; j += blockDim.x) s += expf(xr[j] - m);
  }
  s = block_reduce_sum(s, red);
  const float inv = 1.0f / s;

  if (vec) {
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    float4* yr4 = reinterpret_cast<float4*>(yr);
    for (int i = threadIdx.x; i < N4; i += blockDim.x) {
      float4 v = xr4[i];
      float4 o;
      o.x = expf(v.x - m) * inv;
      o.y = expf(v.y - m) * inv;
      o.z = expf(v.z - m) * inv;
      o.w = expf(v.w - m) * inv;
      yr4[i] = o;
    }
  } else {
    for (int j = threadIdx.x; j < N; j += blockDim.x)
      yr[j] = expf(xr[j] - m) * inv;
  }
}

// ---------------------------------------------------------------------------
// Kernel 4: online softmax (Milakov & Gimelshein 2018), one block per row.
// Tracks a running (max m, sum s) pair per thread and merges pairs with the
// rescaling rule s = s1*exp(m1-M) + s2*exp(m2-M), M = max(m1, m2). The max
// and sum passes fuse into one, so each row is read twice instead of three
// times: a 4/3 traffic reduction over kernel 3. This is the same trick that
// makes FlashAttention possible.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void online_update(float& m, float& s, float v) {
  float mn = fmaxf(m, v);
  s = s * expf(m - mn) + expf(v - mn);
  m = mn;
}

__device__ __forceinline__ void online_merge(float& m, float& s, float m2,
                                             float s2) {
  float mn = fmaxf(m, m2);
  s = s * expf(m - mn) + s2 * expf(m2 - mn);
  m = mn;
}

__device__ __forceinline__ void warp_merge_ms(float& m, float& s) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float m2 = __shfl_xor_sync(0xffffffffu, m, offset);
    float s2 = __shfl_xor_sync(0xffffffffu, s, offset);
    online_merge(m, s, m2, s2);
  }
}

// smem must hold 64 floats; every thread receives the merged (m, s).
__device__ __forceinline__ void block_merge_ms(float& m, float& s,
                                               float* smem) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  warp_merge_ms(m, s);
  if (lane == 0) {
    smem[wid] = m;
    smem[32 + wid] = s;
  }
  __syncthreads();
  if (wid == 0) {
    float mm = (lane < nwarps) ? smem[lane] : -FLT_MAX;
    float ss = (lane < nwarps) ? smem[32 + lane] : 0.0f;
    warp_merge_ms(mm, ss);
    if (lane == 0) {
      smem[0] = mm;
      smem[32] = ss;
    }
  }
  __syncthreads();
  m = smem[0];
  s = smem[32];
  __syncthreads();
}

__global__ void softmax_k4_online_vec4(const float* __restrict__ x,
                                       float* __restrict__ y, int M, int N) {
  __shared__ float red[64];
  const int row = blockIdx.x;
  if (row >= M) return;
  const float* xr = x + (size_t)row * N;
  float* yr = y + (size_t)row * N;
  const bool vec = (N & 3) == 0;
  const int N4 = N >> 2;

  // Pass 1: fused max + sum.
  float m = -FLT_MAX, s = 0.0f;
  if (vec) {
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    for (int i = threadIdx.x; i < N4; i += blockDim.x) {
      float4 v = xr4[i];
      online_update(m, s, v.x);
      online_update(m, s, v.y);
      online_update(m, s, v.z);
      online_update(m, s, v.w);
    }
  } else {
    for (int j = threadIdx.x; j < N; j += blockDim.x)
      online_update(m, s, xr[j]);
  }
  block_merge_ms(m, s, red);
  const float inv = 1.0f / s;

  // Pass 2: normalize and write.
  if (vec) {
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    float4* yr4 = reinterpret_cast<float4*>(yr);
    for (int i = threadIdx.x; i < N4; i += blockDim.x) {
      float4 v = xr4[i];
      float4 o;
      o.x = expf(v.x - m) * inv;
      o.y = expf(v.y - m) * inv;
      o.z = expf(v.z - m) * inv;
      o.w = expf(v.w - m) * inv;
      yr4[i] = o;
    }
  } else {
    for (int j = threadIdx.x; j < N; j += blockDim.x)
      yr[j] = expf(xr[j] - m) * inv;
  }
}

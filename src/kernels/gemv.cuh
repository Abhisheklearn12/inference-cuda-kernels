#pragma once
// GEMV: y[M] = A[M,K] @ x[K], A row-major, fp32.
//
// This is the core op of single-batch LLM decoding: every weight matrix is
// applied to one activation vector per generated token. Arithmetic intensity
// is ~0.5 FLOP/byte, so GEMV is memory bandwidth bound. On an RTX 3060 the
// ceiling is 360 GB/s, which for fp32 means at most ~180 GFLOPS regardless
// of how much compute the kernel uses. The progression below is therefore
// about reaching peak bandwidth, not peak FLOPS.

#include "../common.cuh"

// ---------------------------------------------------------------------------
// Kernel 1: naive, one thread per row.
// Each thread walks one row sequentially. At any instant the 32 threads of a
// warp read 32 different rows at the same column, so consecutive addresses
// within a warp are K*4 bytes apart: every load is an uncoalesced 32-byte
// sector of which only 4 bytes are used.
// ---------------------------------------------------------------------------
__global__ void gemv_k1_naive(const float* A, const float* x, float* y,
                              int M, int K) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;
  const float* a = A + (size_t)row * K;
  float acc = 0.0f;
  for (int k = 0; k < K; k++) acc += a[k] * x[k];
  y[row] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 2: one warp per row, coalesced loads, warp shuffle reduction.
// Lane l reads columns l, l+32, l+64, ... so each warp load is one fully
// used 128-byte transaction. Partial sums are combined with shuffles, no
// shared memory needed.
// ---------------------------------------------------------------------------
__global__ void gemv_k2_warp(const float* __restrict__ A,
                             const float* __restrict__ x,
                             float* __restrict__ y, int M, int K) {
  const int warps_per_block = blockDim.x >> 5;
  const int row = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= M) return;
  const float* a = A + (size_t)row * K;
  float acc = 0.0f;
  for (int k = lane; k < K; k += 32) acc += a[k] * x[k];
  acc = warp_reduce_sum(acc);
  if (lane == 0) y[row] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 3: one warp per row + float4 vectorized loads.
// 128-bit loads quarter the instruction count per byte and help the memory
// pipeline issue maximal-width transactions. Falls back to the scalar path
// when K is not a multiple of 4 (row pointers would be misaligned).
// ---------------------------------------------------------------------------
__global__ void gemv_k3_vec4(const float* __restrict__ A,
                             const float* __restrict__ x,
                             float* __restrict__ y, int M, int K) {
  const int warps_per_block = blockDim.x >> 5;
  const int row = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= M) return;
  const float* a = A + (size_t)row * K;
  float acc = 0.0f;
  if ((K & 3) == 0) {
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int K4 = K >> 2;
    for (int i = lane; i < K4; i += 32) {
      float4 av = a4[i];
      float4 xv = x4[i];
      acc += av.x * xv.x + av.y * xv.y + av.z * xv.z + av.w * xv.w;
    }
  } else {
    for (int k = lane; k < K; k += 32) acc += a[k] * x[k];
  }
  acc = warp_reduce_sum(acc);
  if (lane == 0) y[row] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 4: one block (256 threads) per row, float4, hierarchical reduction.
// More threads per row shortens each thread's loop; useful when K is large
// or M is small (few rows means few warps in kernel 3, hurting occupancy).
// Costs a shared-memory cross-warp reduction per row.
// ---------------------------------------------------------------------------
__global__ void gemv_k4_block(const float* __restrict__ A,
                              const float* __restrict__ x,
                              float* __restrict__ y, int M, int K) {
  __shared__ float red[32];
  const int row = blockIdx.x;
  if (row >= M) return;
  const float* a = A + (size_t)row * K;
  float acc = 0.0f;
  if ((K & 3) == 0) {
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int K4 = K >> 2;
    for (int i = threadIdx.x; i < K4; i += blockDim.x) {
      float4 av = a4[i];
      float4 xv = x4[i];
      acc += av.x * xv.x + av.y * xv.y + av.z * xv.z + av.w * xv.w;
    }
  } else {
    for (int k = threadIdx.x; k < K; k += blockDim.x) acc += a[k] * x[k];
  }
  acc = block_reduce_sum(acc, red);
  if (threadIdx.x == 0) y[row] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 5: kernel 3 plus tuning: __ldg through the read-only cache, 4x
// unrolling so more loads are in flight per thread, __launch_bounds__ to
// pin occupancy, and a grid-stride loop over rows so the grid size can be
// decoupled from M. Gains over kernel 3 are small because kernel 3 is
// already near the bandwidth roofline; past that point every further trick
// hits diminishing returns.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(128, 8) void gemv_k5_tuned(
    const float* __restrict__ A, const float* __restrict__ x,
    float* __restrict__ y, int M, int K) {
  const int warps_per_block = blockDim.x >> 5;
  const int lane = threadIdx.x & 31;
  const int gwarp = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  const int nwarps = gridDim.x * warps_per_block;
  if ((K & 3) == 0) {
    const int K4 = K >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int row = gwarp; row < M; row += nwarps) {
      const float4* a4 = reinterpret_cast<const float4*>(A + (size_t)row * K);
      float acc = 0.0f;
#pragma unroll 4
      for (int i = lane; i < K4; i += 32) {
        float4 av = __ldg(&a4[i]);
        float4 xv = __ldg(&x4[i]);
        acc += av.x * xv.x + av.y * xv.y + av.z * xv.z + av.w * xv.w;
      }
      acc = warp_reduce_sum(acc);
      if (lane == 0) y[row] = acc;
    }
  } else {
    for (int row = gwarp; row < M; row += nwarps) {
      const float* a = A + (size_t)row * K;
      float acc = 0.0f;
      for (int k = lane; k < K; k += 32) acc += __ldg(&a[k]) * __ldg(&x[k]);
      acc = warp_reduce_sum(acc);
      if (lane == 0) y[row] = acc;
    }
  }
}

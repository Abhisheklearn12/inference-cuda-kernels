#pragma once
// RMSNorm on a [M, N] matrix with weight w[N], fp32 (Llama-family norm):
//   y[i,j] = x[i,j] * rsqrt(mean_j(x[i,j]^2) + eps) * w[j]
//
// Memory bound like everything else at decode time. Two passes over the row
// are required (reduce, then normalize) since a row does not fit in
// registers for realistic N.

#include "../common.cuh"

// ---------------------------------------------------------------------------
// Kernel 1: naive, one thread per row.
// ---------------------------------------------------------------------------
__global__ void rmsnorm_k1_naive(const float* x, const float* w, float* y,
                                 int M, int N, float eps) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;
  const float* xr = x + (size_t)row * N;
  float* yr = y + (size_t)row * N;
  float ss = 0.0f;
  for (int j = 0; j < N; j++) ss += xr[j] * xr[j];
  float inv = rsqrtf(ss / N + eps);
  for (int j = 0; j < N; j++) yr[j] = xr[j] * inv * w[j];
}

// ---------------------------------------------------------------------------
// Kernel 2: one block (256 threads) per row, coalesced, block reduction.
// ---------------------------------------------------------------------------
__global__ void rmsnorm_k2_block(const float* __restrict__ x,
                                 const float* __restrict__ w,
                                 float* __restrict__ y, int M, int N,
                                 float eps) {
  __shared__ float red[32];
  const int row = blockIdx.x;
  if (row >= M) return;
  const float* xr = x + (size_t)row * N;
  float* yr = y + (size_t)row * N;
  float ss = 0.0f;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    float v = xr[j];
    ss += v * v;
  }
  ss = block_reduce_sum(ss, red);
  const float inv = rsqrtf(ss / N + eps);
  for (int j = threadIdx.x; j < N; j += blockDim.x)
    yr[j] = xr[j] * inv * w[j];
}

// ---------------------------------------------------------------------------
// Kernel 3: kernel 2 + float4 vectorized loads and stores.
// Falls back to the scalar path when N is not a multiple of 4.
// ---------------------------------------------------------------------------
__global__ void rmsnorm_k3_block_vec4(const float* __restrict__ x,
                                      const float* __restrict__ w,
                                      float* __restrict__ y, int M, int N,
                                      float eps) {
  __shared__ float red[32];
  const int row = blockIdx.x;
  if (row >= M) return;
  const float* xr = x + (size_t)row * N;
  float* yr = y + (size_t)row * N;
  const bool vec = (N & 3) == 0;
  const int N4 = N >> 2;

  float ss = 0.0f;
  if (vec) {
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    for (int i = threadIdx.x; i < N4; i += blockDim.x) {
      float4 v = xr4[i];
      ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  } else {
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      float v = xr[j];
      ss += v * v;
    }
  }
  ss = block_reduce_sum(ss, red);
  const float inv = rsqrtf(ss / N + eps);

  if (vec) {
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* yr4 = reinterpret_cast<float4*>(yr);
    for (int i = threadIdx.x; i < N4; i += blockDim.x) {
      float4 v = xr4[i];
      float4 g = w4[i];
      float4 o;
      o.x = v.x * inv * g.x;
      o.y = v.y * inv * g.y;
      o.z = v.z * inv * g.z;
      o.w = v.w * inv * g.w;
      yr4[i] = o;
    }
  } else {
    for (int j = threadIdx.x; j < N; j += blockDim.x)
      yr[j] = xr[j] * inv * w[j];
  }
}

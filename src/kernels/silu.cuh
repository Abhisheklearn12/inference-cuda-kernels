#pragma once
// SiLU (swish) elementwise activation, fp32: y = x * sigmoid(x) = x / (1 + exp(-x))
//
// The MLP activation in Llama-family models. Purely memory bound (1 read +
// 1 write per element, a few FLOPs). A perfect kernel moves 8 bytes per
// element at DRAM speed; the progression shows how little is needed to get
// there and how to measure it.

#include "../common.cuh"

__device__ __forceinline__ float silu(float v) {
  return v / (1.0f + expf(-v));
}

// ---------------------------------------------------------------------------
// Kernel 1: naive, one thread per element, exactly one grid launch that
// covers n. Already coalesced because thread i touches element i.
// ---------------------------------------------------------------------------
__global__ void silu_k1_naive(const float* x, float* y, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) y[i] = silu(x[i]);
}

// ---------------------------------------------------------------------------
// Kernel 2: grid-stride loop. The grid size is decoupled from n (a fixed
// number of blocks each processes many elements), which amortizes block
// scheduling overhead and lets one binary handle any n.
// ---------------------------------------------------------------------------
__global__ void silu_k2_gridstride(const float* __restrict__ x,
                                   float* __restrict__ y, long long n) {
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
       i += stride)
    y[i] = silu(x[i]);
}

// ---------------------------------------------------------------------------
// Kernel 3: grid-stride + float4. 128-bit accesses, one quarter the memory
// instructions. The scalar tail (n % 4 elements) is handled by the first
// few threads of the grid.
// ---------------------------------------------------------------------------
__global__ void silu_k3_vec4(const float* __restrict__ x,
                             float* __restrict__ y, long long n) {
  const long long n4 = n >> 2;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float4* y4 = reinterpret_cast<float4*>(y);
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = tid; i < n4; i += stride) {
    float4 v = x4[i];
    float4 o;
    o.x = silu(v.x);
    o.y = silu(v.y);
    o.z = silu(v.z);
    o.w = silu(v.w);
    y4[i] = o;
  }
  const long long tail = n4 << 2;
  const long long i = tail + tid;
  if (i < n) y[i] = silu(x[i]);
}

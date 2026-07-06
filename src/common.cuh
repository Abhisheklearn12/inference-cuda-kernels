#pragma once
// Shared helpers: error checking, reductions, RNG, verification, benchmarking.
// Target: NVIDIA RTX 3060 (Ampere GA106, sm_86), but nothing here is 3060-specific.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err_));                                      \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Device-side reductions
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
  return v;
}

// smem must hold at least 32 floats. Every thread receives the result.
// The trailing __syncthreads() makes back-to-back calls on the same smem safe.
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  if (wid == 0) {
    float w = (lane < nwarps) ? smem[lane] : 0.0f;
    w = warp_reduce_sum(w);
    if (lane == 0) smem[0] = w;
  }
  __syncthreads();
  float out = smem[0];
  __syncthreads();
  return out;
}

__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  v = warp_reduce_max(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  if (wid == 0) {
    float w = (lane < nwarps) ? smem[lane] : -FLT_MAX;
    w = warp_reduce_max(w);
    if (lane == 0) smem[0] = w;
  }
  __syncthreads();
  float out = smem[0];
  __syncthreads();
  return out;
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

inline void fill_randn(float* p, size_t n, unsigned int seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < n; i++) p[i] = dist(gen);
}

struct CheckResult {
  bool pass;
  double max_abs;   // max absolute error
  double max_rel;   // max relative error (only where |ref| > 1e-6)
  size_t first_bad; // index of first failing element, only valid if !pass
};

// Element i passes if |got - want| <= atol + rtol * |want|.
inline CheckResult check_close(const float* got, const float* want, size_t n,
                               double atol, double rtol) {
  CheckResult r{true, 0.0, 0.0, 0};
  for (size_t i = 0; i < n; i++) {
    double g = got[i], w = want[i];
    if (std::isnan(g) || std::isinf(g)) {
      if (r.pass) { r.pass = false; r.first_bad = i; }
      r.max_abs = INFINITY;
      continue;
    }
    double abs_err = std::fabs(g - w);
    if (abs_err > r.max_abs) r.max_abs = abs_err;
    if (std::fabs(w) > 1e-6) {
      double rel = abs_err / std::fabs(w);
      if (rel > r.max_rel) r.max_rel = rel;
    }
    if (abs_err > atol + rtol * std::fabs(w)) {
      if (r.pass) { r.pass = false; r.first_bad = i; }
    }
  }
  return r;
}

// Times f() with CUDA events. Runs a few warmups, sizes the iteration count so
// the timed region is roughly 250 ms, and returns average milliseconds.
template <typename F>
inline float bench_ms(F&& f) {
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));

  f();
  f();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(t0));
  f();
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float single_ms;
  CUDA_CHECK(cudaEventElapsedTime(&single_ms, t0, t1));

  int iters = (int)(250.0f / (single_ms > 1e-4f ? single_ms : 1e-4f));
  if (iters < 3) iters = 3;
  if (iters > 1000) iters = 1000;

  CUDA_CHECK(cudaEventRecord(t0));
  for (int i = 0; i < iters; i++) f();
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float total_ms;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, t0, t1));

  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
  return total_ms / iters;
}

// Theoretical peak DRAM bandwidth in GB/s from device properties.
// GDDR6 is double data rate: 2 * memory clock * bus width in bytes.
inline double peak_bandwidth_gbs() {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  return 2.0 * (double)p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;
}

inline void print_device_banner() {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  printf("GPU: %s (sm_%d%d, %d SMs, peak DRAM bandwidth %.1f GB/s)\n\n",
         p.name, p.major, p.minor, p.multiProcessorCount,
         peak_bandwidth_gbs());
}

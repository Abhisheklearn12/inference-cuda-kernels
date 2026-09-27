// Prints the real hardware numbers of the installed GPU, plus the derived
// theoretical peaks used as rooflines by the benchmarks.

#include "common.cuh"

int main() {
  int dev = 0;
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

  // Clocks are queried as attributes: CUDA 13 removed them from cudaDeviceProp.
  int clock_khz = 0, mem_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev));
  double core_ghz = clock_khz * 1e-6;             // boost clock
  double mem_ghz = mem_khz * 1e-6;                // half the effective rate
  // GA10x (sm_86) has 128 fp32 CUDA cores per SM.
  int cores_per_sm = 128;
  int cores = cores_per_sm * p.multiProcessorCount;
  double peak_fp32 = 2.0 * cores * core_ghz;      // GFLOPS (FMA = 2 FLOP)
  double peak_bw = 2.0 * mem_ghz * (p.memoryBusWidth / 8.0);  // GB/s

  printf("Device %d: %s\n", dev, p.name);
  printf("  Compute capability:        sm_%d%d\n", p.major, p.minor);
  printf("  SMs:                       %d\n", p.multiProcessorCount);
  printf("  FP32 cores per SM:         %d (GA10x)\n", cores_per_sm);
  printf("  FP32 cores total:          %d\n", cores);
  printf("  Boost clock:               %.0f MHz\n", core_ghz * 1e3);
  printf("  Peak FP32 (FMA):           %.1f GFLOPS = %.2f TFLOPS\n",
         peak_fp32, peak_fp32 * 1e-3);
  printf("  Memory clock:              %.0f MHz (%.1f Gbps effective)\n",
         mem_ghz * 1e3, 2.0 * mem_ghz);
  printf("  Memory bus width:          %d bit\n", p.memoryBusWidth);
  printf("  Peak DRAM bandwidth:       %.1f GB/s\n", peak_bw);
  printf("  VRAM:                      %.1f GiB\n",
         p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("  L2 cache:                  %d KiB\n", p.l2CacheSize / 1024);
  printf("  Shared memory per SM:      %zu KiB\n",
         p.sharedMemPerMultiprocessor / 1024);
  printf("  Shared memory per block:   %zu KiB (%zu KiB with opt-in)\n",
         p.sharedMemPerBlock / 1024, p.sharedMemPerBlockOptin / 1024);
  printf("  Registers per SM:          %d\n", p.regsPerMultiprocessor);
  printf("  Max threads per SM:        %d\n", p.maxThreadsPerMultiProcessor);
  printf("  Max threads per block:     %d\n", p.maxThreadsPerBlock);
  printf("  Warp size:                 %d\n", p.warpSize);
  printf("\nRoofline crossover: %.1f FLOP/byte. Every kernel in this repo\n",
         peak_fp32 / peak_bw);
  printf("sits far below that (GEMV ~0.5), so DRAM bandwidth is the target.\n");
  return 0;
}

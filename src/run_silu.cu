// SiLU runner: correctness vs double-precision CPU reference, benchmark.
// Usage: ./silu [kernel_id]   (0 or no arg = all kernels)

#include <vector>
#include "common.cuh"
#include "kernels/silu.cuh"

constexpr int NUM_KERNELS = 3;
constexpr int GRID_BLOCKS = 2048;  // for grid-stride kernels

static void launch(int id, const float* x, float* y, long long n) {
  switch (id) {
    case 1: {
      long long grid = (n + 255) / 256;
      silu_k1_naive<<<(unsigned)grid, 256>>>(x, y, n);
      break;
    }
    case 2:
      silu_k2_gridstride<<<GRID_BLOCKS, 256>>>(x, y, n);
      break;
    case 3: {
      // One float4 per thread. A small fixed grid (like kernel 2) makes each
      // thread stride 8 MB between iterations, which measurably hurts DRAM
      // locality on GA106; covering n4 directly keeps accesses dense.
      long long n4 = n >> 2;
      long long grid = (n4 + 255) / 256;
      if (grid < 1) grid = 1;
      silu_k3_vec4<<<(unsigned)grid, 256>>>(x, y, n);
      break;
    }
    default:
      fprintf(stderr, "bad kernel id %d\n", id);
      exit(1);
  }
  CUDA_CHECK(cudaGetLastError());
}

static void silu_ref(const float* x, float* y, long long n) {
  for (long long i = 0; i < n; i++) {
    double v = x[i];
    y[i] = (float)(v / (1.0 + std::exp(-v)));
  }
}

int main(int argc, char** argv) {
  int which = (argc > 1) ? atoi(argv[1]) : 0;
  print_device_banner();

  std::vector<int> ids;
  if (which == 0)
    for (int i = 1; i <= NUM_KERNELS; i++) ids.push_back(i);
  else
    ids.push_back(which);

  const long long check_sizes[] = {1, 5, 1023, 4097, (1 << 20) + 3};
  const long long bench_n = 1LL << 26;  // 64M elements, 512 MB of traffic
  const double atol = 1e-5, rtol = 1e-5;

  float* hx = (float*)malloc(bench_n * sizeof(float));
  float* hy = (float*)malloc(bench_n * sizeof(float));
  float* href = (float*)malloc(bench_n * sizeof(float));
  fill_randn(hx, bench_n, 6);

  float *dx, *dy;
  CUDA_CHECK(cudaMalloc(&dx, bench_n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, bench_n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx, bench_n * sizeof(float),
                        cudaMemcpyHostToDevice));

  bool all_pass = true;
  printf("== correctness (atol %.0e, rtol %.0e, vs fp64 CPU reference) ==\n",
         atol, rtol);
  for (long long n : check_sizes) {
    silu_ref(hx, href, n);
    for (int id : ids) {
      CUDA_CHECK(cudaMemset(dy, 0xFF, (size_t)n * sizeof(float)));
      launch(id, dx, dy, n);
      CUDA_CHECK(cudaMemcpy(hy, dy, (size_t)n * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CheckResult r = check_close(hy, href, (size_t)n, atol, rtol);
      printf("  kernel %d  n=%9lld  %s  (max abs %.2e, max rel %.2e)\n", id, n,
             r.pass ? "PASS" : "FAIL", r.max_abs, r.max_rel);
      all_pass &= r.pass;
    }
  }

  const double peak = peak_bandwidth_gbs();
  printf("\n== benchmark (n = %lld) ==\n", bench_n);
  silu_ref(hx, href, bench_n);
  for (int id : ids) {
    launch(id, dx, dy, bench_n);
    CUDA_CHECK(cudaMemcpy(hy, dy, bench_n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CheckResult r = check_close(hy, href, (size_t)bench_n, atol, rtol);
    all_pass &= r.pass;
    float ms = bench_ms([&] { launch(id, dx, dy, bench_n); });
    double bytes = 8.0 * (double)bench_n;
    double gbs = bytes / (ms * 1e-3) / 1e9;
    printf("  kernel %d: %8.3f ms  %7.1f GB/s (%5.1f%% of peak)  [%s]\n", id,
           ms, gbs, 100.0 * gbs / peak, r.pass ? "PASS" : "FAIL");
  }

  printf("\n%s\n", all_pass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  free(hx); free(hy); free(href);
  return all_pass ? 0 : 1;
}

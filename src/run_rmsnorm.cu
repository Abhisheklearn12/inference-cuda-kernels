// RMSNorm runner: correctness vs double-precision CPU reference, benchmark.
// Usage: ./rmsnorm [kernel_id]   (0 or no arg = all kernels)

#include <vector>
#include "common.cuh"
#include "kernels/rmsnorm.cuh"

constexpr int NUM_KERNELS = 3;
constexpr float EPS = 1e-5f;

static void launch(int id, const float* x, const float* w, float* y, int M,
                   int N) {
  switch (id) {
    case 1:
      rmsnorm_k1_naive<<<cdiv(M, 256), 256>>>(x, w, y, M, N, EPS);
      break;
    case 2:
      rmsnorm_k2_block<<<M, 256>>>(x, w, y, M, N, EPS);
      break;
    case 3:
      rmsnorm_k3_block_vec4<<<M, 256>>>(x, w, y, M, N, EPS);
      break;
    default:
      fprintf(stderr, "bad kernel id %d\n", id);
      exit(1);
  }
  CUDA_CHECK(cudaGetLastError());
}

static void rmsnorm_ref(const float* x, const float* w, float* y, int M,
                        int N) {
  for (int i = 0; i < M; i++) {
    const float* xr = x + (size_t)i * N;
    float* yr = y + (size_t)i * N;
    double ss = 0.0;
    for (int j = 0; j < N; j++) ss += (double)xr[j] * (double)xr[j];
    double inv = 1.0 / std::sqrt(ss / N + (double)EPS);
    for (int j = 0; j < N; j++) yr[j] = (float)((double)xr[j] * inv * w[j]);
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

  struct Shape { int M, N; };
  const Shape check_shapes[] = {{1, 1},     {3, 7},    {17, 129},
                                {100, 999}, {5, 4096}, {4096, 4096}};
  // 4096x4096: prefill-like batch of rows. 1x4096: single decode token,
  // the latency-critical case.
  const Shape bench_shapes[] = {{4096, 4096}, {1, 4096}};
  const double atol = 1e-3, rtol = 1e-3;
  const size_t maxN = (size_t)4096 * 4096;
  const int maxCols = 4096;

  float* hx = (float*)malloc(maxN * sizeof(float));
  float* hw = (float*)malloc(maxCols * sizeof(float));
  float* hy = (float*)malloc(maxN * sizeof(float));
  float* href = (float*)malloc(maxN * sizeof(float));
  fill_randn(hx, maxN, 4);
  fill_randn(hw, maxCols, 5);

  float *dx, *dw, *dy;
  CUDA_CHECK(cudaMalloc(&dx, maxN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dw, maxCols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, maxN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx, maxN * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, hw, maxCols * sizeof(float),
                        cudaMemcpyHostToDevice));

  bool all_pass = true;
  printf("== correctness (atol %.0e, rtol %.0e, vs fp64 CPU reference) ==\n",
         atol, rtol);
  for (const Shape& s : check_shapes) {
    size_t n = (size_t)s.M * s.N;
    rmsnorm_ref(hx, hw, href, s.M, s.N);
    for (int id : ids) {
      CUDA_CHECK(cudaMemset(dy, 0xFF, n * sizeof(float)));
      launch(id, dx, dw, dy, s.M, s.N);
      CUDA_CHECK(cudaMemcpy(hy, dy, n * sizeof(float), cudaMemcpyDeviceToHost));
      CheckResult r = check_close(hy, href, n, atol, rtol);
      printf("  kernel %d  M=%5d N=%5d  %s  (max abs %.2e, max rel %.2e)\n",
             id, s.M, s.N, r.pass ? "PASS" : "FAIL", r.max_abs, r.max_rel);
      all_pass &= r.pass;
    }
  }

  const double peak = peak_bandwidth_gbs();
  printf("\n== benchmark (effective bytes = 1 read + 1 write per element + w) ==\n");
  for (const Shape& s : bench_shapes) {
    size_t n = (size_t)s.M * s.N;
    rmsnorm_ref(hx, hw, href, s.M, s.N);
    printf("  M=%5d N=%5d\n", s.M, s.N);
    for (int id : ids) {
      launch(id, dx, dw, dy, s.M, s.N);
      CUDA_CHECK(cudaMemcpy(hy, dy, n * sizeof(float), cudaMemcpyDeviceToHost));
      CheckResult r = check_close(hy, href, n, atol, rtol);
      all_pass &= r.pass;
      float ms = bench_ms([&] { launch(id, dx, dw, dy, s.M, s.N); });
      double bytes = 4.0 * (2.0 * n + s.N);
      double gbs = bytes / (ms * 1e-3) / 1e9;
      printf("    kernel %d: %8.3f ms  %7.1f GB/s (%5.1f%% of peak)  [%s]\n",
             id, ms, gbs, 100.0 * gbs / peak, r.pass ? "PASS" : "FAIL");
    }
  }

  printf("\n%s\n", all_pass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dw));
  CUDA_CHECK(cudaFree(dy));
  free(hx); free(hw); free(hy); free(href);
  return all_pass ? 0 : 1;
}

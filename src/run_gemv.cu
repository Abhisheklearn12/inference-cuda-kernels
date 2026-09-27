// GEMV runner: correctness vs double-precision CPU reference, then benchmark.
// Usage: ./gemv [kernel_id]   (0 or no arg = all kernels)

#include <vector>
#include "common.cuh"
#include "kernels/gemv.cuh"

static const char* kernel_names[] = {
    "",
    "1: naive, one thread per row (uncoalesced)",
    "2: one warp per row + shuffle reduction (coalesced)",
    "3: one warp per row + float4 loads",
    "4: one block per row + float4 + block reduction",
    "5: warp per row + float4 + ldg + unroll + launch_bounds",
};
constexpr int NUM_KERNELS = 5;

static void launch(int id, const float* A, const float* x, float* y, int M,
                   int K) {
  switch (id) {
    case 1:
      gemv_k1_naive<<<cdiv(M, 256), 256>>>(A, x, y, M, K);
      break;
    case 2:
      gemv_k2_warp<<<cdiv(M, 8), 256>>>(A, x, y, M, K);
      break;
    case 3:
      gemv_k3_vec4<<<cdiv(M, 8), 256>>>(A, x, y, M, K);
      break;
    case 4:
      gemv_k4_block<<<M, 256>>>(A, x, y, M, K);
      break;
    case 5: {
      int grid = cdiv(M, 4);
      if (grid > 65535) grid = 65535;
      gemv_k5_tuned<<<grid, 128>>>(A, x, y, M, K);
      break;
    }
    default:
      fprintf(stderr, "bad kernel id %d\n", id);
      exit(1);
  }
  CUDA_CHECK(cudaGetLastError());
}

static void gemv_ref(const float* A, const float* x, float* y, int M, int K) {
  for (int m = 0; m < M; m++) {
    double acc = 0.0;
    const float* a = A + (size_t)m * K;
    for (int k = 0; k < K; k++) acc += (double)a[k] * (double)x[k];
    y[m] = (float)acc;
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
  for (int id : ids) printf("kernel %s\n", kernel_names[id]);
  printf("\n");

  // Shapes: correctness set includes edge cases (K not divisible by 4, tiny
  // sizes); benchmark set uses Llama-7B-like layer shapes.
  struct Shape { int M, K; };
  const Shape check_shapes[] = {{1, 1},     {7, 3},       {37, 511},
                                {255, 257}, {129, 4096},  {1000, 1003},
                                {4096, 4096}};
  const Shape bench_shapes[] = {{4096, 4096}, {11008, 4096}, {4096, 11008},
                                {32000, 4096}};
  const double atol = 5e-3, rtol = 2e-3;
  const size_t maxA = (size_t)32000 * 11008;  // covers every shape above

  float* hA = (float*)malloc(maxA * sizeof(float));
  float* hx = (float*)malloc(11008 * sizeof(float));
  float* hy = (float*)malloc(32000 * sizeof(float));
  float* href = (float*)malloc(32000 * sizeof(float));
  fill_randn(hA, maxA, 1);
  fill_randn(hx, 11008, 2);

  float *dA, *dx, *dy;
  CUDA_CHECK(cudaMalloc(&dA, maxA * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dx, 11008 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, 32000 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA, maxA * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dx, hx, 11008 * sizeof(float), cudaMemcpyHostToDevice));

  // Correctness.
  bool all_pass = true;
  printf("== correctness (atol %.0e, rtol %.0e, vs fp64 CPU reference) ==\n",
         atol, rtol);
  for (const Shape& s : check_shapes) {
    gemv_ref(hA, hx, href, s.M, s.K);
    for (int id : ids) {
      CUDA_CHECK(cudaMemset(dy, 0xFF, (size_t)s.M * sizeof(float)));
      launch(id, dA, dx, dy, s.M, s.K);
      CUDA_CHECK(cudaMemcpy(hy, dy, (size_t)s.M * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CheckResult r = check_close(hy, href, s.M, atol, rtol);
      printf("  kernel %d  M=%5d K=%5d  %s  (max abs %.2e, max rel %.2e)\n",
             id, s.M, s.K, r.pass ? "PASS" : "FAIL", r.max_abs, r.max_rel);
      all_pass &= r.pass;
    }
  }

  // Benchmark (each bench shape is also verified once per kernel).
  const double peak = peak_bandwidth_gbs();
  printf("\n== benchmark ==\n");
  for (const Shape& s : bench_shapes) {
    gemv_ref(hA, hx, href, s.M, s.K);
    printf("  M=%5d K=%5d\n", s.M, s.K);
    for (int id : ids) {
      CUDA_CHECK(cudaMemset(dy, 0xFF, (size_t)s.M * sizeof(float)));
      launch(id, dA, dx, dy, s.M, s.K);
      CUDA_CHECK(cudaMemcpy(hy, dy, (size_t)s.M * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CheckResult r = check_close(hy, href, s.M, atol, rtol);
      all_pass &= r.pass;
      float ms = bench_ms([&] { launch(id, dA, dx, dy, s.M, s.K); });
      double bytes = 4.0 * ((double)s.M * s.K + s.K + s.M);
      double gbs = bytes / (ms * 1e-3) / 1e9;
      double gflops = 2.0 * s.M * s.K / (ms * 1e-3) / 1e9;
      printf("    kernel %d: %8.3f ms  %7.1f GB/s (%5.1f%% of peak)  "
             "%6.1f GFLOPS  [%s]\n",
             id, ms, gbs, 100.0 * gbs / peak, gflops,
             r.pass ? "PASS" : "FAIL");
    }
  }

  printf("\n%s\n", all_pass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  free(hA); free(hx); free(hy); free(href);
  return all_pass ? 0 : 1;
}

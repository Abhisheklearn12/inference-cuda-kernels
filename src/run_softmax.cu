// Softmax runner: correctness vs double-precision CPU reference, benchmark.
// Usage: ./softmax [kernel_id]   (0 or no arg = all kernels)

#include <vector>
#include "common.cuh"
#include "kernels/softmax.cuh"

static const char* kernel_names[] = {
    "",
    "1: naive, one thread per row, 3 passes",
    "2: one warp per row + shuffle reductions, 3 passes",
    "3: one block per row + float4, 3 passes",
    "4: online softmax (fused max+sum), block per row + float4, 2 passes",
};
constexpr int NUM_KERNELS = 4;

static void launch(int id, const float* x, float* y, int M, int N) {
  switch (id) {
    case 1:
      softmax_k1_naive<<<cdiv(M, 256), 256>>>(x, y, M, N);
      break;
    case 2:
      softmax_k2_warp<<<cdiv(M, 8), 256>>>(x, y, M, N);
      break;
    case 3:
      softmax_k3_block_vec4<<<M, 256>>>(x, y, M, N);
      break;
    case 4:
      softmax_k4_online_vec4<<<M, 256>>>(x, y, M, N);
      break;
    default:
      fprintf(stderr, "bad kernel id %d\n", id);
      exit(1);
  }
  CUDA_CHECK(cudaGetLastError());
}

static void softmax_ref(const float* x, float* y, int M, int N) {
  for (int i = 0; i < M; i++) {
    const float* xr = x + (size_t)i * N;
    float* yr = y + (size_t)i * N;
    double m = -INFINITY;
    for (int j = 0; j < N; j++) m = std::max(m, (double)xr[j]);
    double s = 0.0;
    for (int j = 0; j < N; j++) s += std::exp((double)xr[j] - m);
    for (int j = 0; j < N; j++) yr[j] = (float)(std::exp((double)xr[j] - m) / s);
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

  struct Shape { int M, N; };
  const Shape check_shapes[] = {{1, 1},     {3, 7},      {17, 129},
                                {100, 999}, {5, 4096},   {129, 1000},
                                {4096, 4096}};
  // 4096x4096: attention-score-like grid. 32x32000: batch of logit rows
  // (Llama vocab), where per-row parallelism decides everything.
  const Shape bench_shapes[] = {{4096, 4096}, {32, 32000}};
  const double atol = 1e-5, rtol = 1e-3;
  const size_t maxN = (size_t)4096 * 4096;  // >= 32*32000 too

  float* hx = (float*)malloc(maxN * sizeof(float));
  float* hy = (float*)malloc(maxN * sizeof(float));
  float* href = (float*)malloc(maxN * sizeof(float));
  fill_randn(hx, maxN, 3);

  float *dx, *dy;
  CUDA_CHECK(cudaMalloc(&dx, maxN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, maxN * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx, maxN * sizeof(float), cudaMemcpyHostToDevice));

  bool all_pass = true;
  printf("== correctness (atol %.0e, rtol %.0e, vs fp64 CPU reference) ==\n",
         atol, rtol);
  for (const Shape& s : check_shapes) {
    size_t n = (size_t)s.M * s.N;
    softmax_ref(hx, href, s.M, s.N);
    for (int id : ids) {
      CUDA_CHECK(cudaMemset(dy, 0xFF, n * sizeof(float)));
      launch(id, dx, dy, s.M, s.N);
      CUDA_CHECK(cudaMemcpy(hy, dy, n * sizeof(float), cudaMemcpyDeviceToHost));
      CheckResult r = check_close(hy, href, n, atol, rtol);
      printf("  kernel %d  M=%5d N=%5d  %s  (max abs %.2e, max rel %.2e)\n",
             id, s.M, s.N, r.pass ? "PASS" : "FAIL", r.max_abs, r.max_rel);
      all_pass &= r.pass;
    }
  }

  const double peak = peak_bandwidth_gbs();
  printf("\n== benchmark (effective bytes = 1 read + 1 write per element) ==\n");
  for (const Shape& s : bench_shapes) {
    size_t n = (size_t)s.M * s.N;
    softmax_ref(hx, href, s.M, s.N);
    printf("  M=%5d N=%5d\n", s.M, s.N);
    for (int id : ids) {
      launch(id, dx, dy, s.M, s.N);
      CUDA_CHECK(cudaMemcpy(hy, dy, n * sizeof(float), cudaMemcpyDeviceToHost));
      CheckResult r = check_close(hy, href, n, atol, rtol);
      all_pass &= r.pass;
      float ms = bench_ms([&] { launch(id, dx, dy, s.M, s.N); });
      double bytes = 8.0 * (double)n;
      double gbs = bytes / (ms * 1e-3) / 1e9;
      printf("    kernel %d: %8.3f ms  %7.1f GB/s (%5.1f%% of peak)  [%s]\n",
             id, ms, gbs, 100.0 * gbs / peak, r.pass ? "PASS" : "FAIL");
    }
  }

  printf("\n%s\n", all_pass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  free(hx); free(hy); free(href);
  return all_pass ? 0 : 1;
}

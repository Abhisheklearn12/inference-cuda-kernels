// Attention decode runner: correctness vs fp64 CPU reference, benchmark.
// Usage: ./attention [kernel_id]   (0 or no arg = all kernels)

#include <vector>
#include "common.cuh"
#include "kernels/attention.cuh"

constexpr int NUM_KERNELS = 3;
constexpr int BLOCK = 128;

// Scratch for kernel 3 partials, sized in main() for the largest shape.
static float *g_pm, *g_ps, *g_pacc;

// Split count for kernel 3: enough blocks to fill the SMs, but chunks no
// smaller than one tile, and at most 32 splits.
static int split_count(int H, int L) {
  int s = 256 / H;
  int max_useful = cdiv(L, ATTN_TILE);
  if (s > max_useful) s = max_useful;
  if (s > 32) s = 32;
  if (s < 1) s = 1;
  return s;
}

// Returns false if the kernel cannot run for this shape (smem limit).
static bool launch(int id, const float* q, const float* K, const float* V,
                   float* out, int H, int L, int D, float scale) {
  switch (id) {
    case 1: {
      size_t smem = (size_t)(L + D + 32) * sizeof(float);
      if (smem > 48 * 1024) return false;  // default dynamic smem limit
      attention_k1_naive<<<H, BLOCK, smem>>>(q, K, V, out, H, L, D, scale);
      break;
    }
    case 2: {
      size_t smem = (size_t)(ATTN_TILE + 2 * D + 32) * sizeof(float);
      attention_k2_flash<<<H, BLOCK, smem>>>(q, K, V, out, H, L, D, scale);
      break;
    }
    case 3: {
      size_t smem = (size_t)(ATTN_TILE + 2 * D + 32) * sizeof(float);
      int S = split_count(H, L);
      dim3 grid(H, S);
      attention_k3_split<<<grid, BLOCK, smem>>>(q, K, V, g_pm, g_ps, g_pacc,
                                                H, L, D, scale, S);
      attention_k3_reduce<<<H, BLOCK>>>(g_pm, g_ps, g_pacc, out, H, D, S);
      break;
    }
    default:
      fprintf(stderr, "bad kernel id %d\n", id);
      exit(1);
  }
  CUDA_CHECK(cudaGetLastError());
  return true;
}

static void attention_ref(const float* q, const float* K, const float* V,
                          float* out, int H, int L, int D, float scale) {
  std::vector<double> sc(L);
  for (int h = 0; h < H; h++) {
    const float* qh = q + (size_t)h * D;
    const float* Kh = K + (size_t)h * L * D;
    const float* Vh = V + (size_t)h * L * D;
    float* oh = out + (size_t)h * D;
    double m = -INFINITY;
    for (int p = 0; p < L; p++) {
      double dot = 0.0;
      for (int d = 0; d < D; d++)
        dot += (double)qh[d] * (double)Kh[(size_t)p * D + d];
      sc[p] = dot * scale;
      m = std::max(m, sc[p]);
    }
    double s = 0.0;
    for (int p = 0; p < L; p++) {
      sc[p] = std::exp(sc[p] - m);
      s += sc[p];
    }
    for (int d = 0; d < D; d++) {
      double acc = 0.0;
      for (int p = 0; p < L; p++) acc += sc[p] * (double)Vh[(size_t)p * D + d];
      oh[d] = (float)(acc / s);
    }
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

  struct Shape { int H, L, D; };
  const Shape check_shapes[] = {{1, 1, 4},    {2, 17, 8},    {2, 100, 63},
                                {3, 333, 64}, {2, 1000, 128}, {4, 2048, 128}};
  // Llama-7B decode: 32 heads, head dim 128. L = 16384 exceeds kernel 1's
  // shared memory limit and is served by kernel 2 only.
  const Shape bench_shapes[] = {{32, 4096, 128}, {32, 16384, 128}};
  const double atol = 2e-3, rtol = 2e-3;
  const int maxH = 32, maxL = 16384, maxD = 128;
  const size_t maxKV = (size_t)maxH * maxL * maxD;

  float* hq = (float*)malloc((size_t)maxH * maxD * sizeof(float));
  float* hK = (float*)malloc(maxKV * sizeof(float));
  float* hV = (float*)malloc(maxKV * sizeof(float));
  float* ho = (float*)malloc((size_t)maxH * maxD * sizeof(float));
  float* href = (float*)malloc((size_t)maxH * maxD * sizeof(float));
  fill_randn(hq, (size_t)maxH * maxD, 7);
  fill_randn(hK, maxKV, 8);
  fill_randn(hV, maxKV, 9);

  float *dq, *dK, *dV, *dout;
  CUDA_CHECK(cudaMalloc(&dq, (size_t)maxH * maxD * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dK, maxKV * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dV, maxKV * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, (size_t)maxH * maxD * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dq, hq, (size_t)maxH * maxD * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, hK, maxKV * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, hV, maxKV * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Kernel 3 partial buffers: at most 32 heads x 32 splits x 128 dims.
  const size_t maxParts = (size_t)32 * 32;
  CUDA_CHECK(cudaMalloc(&g_pm, maxParts * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&g_ps, maxParts * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&g_pacc, maxParts * maxD * sizeof(float)));

  bool all_pass = true;
  printf("== correctness (atol %.0e, rtol %.0e, vs fp64 CPU reference) ==\n",
         atol, rtol);
  for (const Shape& s : check_shapes) {
    float scale = 1.0f / sqrtf((float)s.D);
    size_t n = (size_t)s.H * s.D;
    attention_ref(hq, hK, hV, href, s.H, s.L, s.D, scale);
    for (int id : ids) {
      CUDA_CHECK(cudaMemset(dout, 0xFF, n * sizeof(float)));
      if (!launch(id, dq, dK, dV, dout, s.H, s.L, s.D, scale)) {
        printf("  kernel %d  H=%2d L=%5d D=%3d  SKIP (smem limit)\n", id, s.H,
               s.L, s.D);
        continue;
      }
      CUDA_CHECK(cudaMemcpy(ho, dout, n * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CheckResult r = check_close(ho, href, n, atol, rtol);
      printf("  kernel %d  H=%2d L=%5d D=%3d  %s  (max abs %.2e, max rel %.2e)\n",
             id, s.H, s.L, s.D, r.pass ? "PASS" : "FAIL", r.max_abs,
             r.max_rel);
      all_pass &= r.pass;
    }
  }

  const double peak = peak_bandwidth_gbs();
  printf("\n== benchmark (effective bytes = KV cache + q + out) ==\n");
  for (const Shape& s : bench_shapes) {
    float scale = 1.0f / sqrtf((float)s.D);
    size_t n = (size_t)s.H * s.D;
    attention_ref(hq, hK, hV, href, s.H, s.L, s.D, scale);
    printf("  H=%2d L=%5d D=%3d\n", s.H, s.L, s.D);
    for (int id : ids) {
      CUDA_CHECK(cudaMemset(dout, 0xFF, n * sizeof(float)));
      if (!launch(id, dq, dK, dV, dout, s.H, s.L, s.D, scale)) {
        printf("    kernel %d: SKIP (smem limit, L too large)\n", id);
        continue;
      }
      CUDA_CHECK(cudaMemcpy(ho, dout, n * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CheckResult r = check_close(ho, href, n, atol, rtol);
      all_pass &= r.pass;
      float ms = bench_ms(
          [&] { launch(id, dq, dK, dV, dout, s.H, s.L, s.D, scale); });
      double bytes = 4.0 * (2.0 * s.H * s.L * s.D + 2.0 * s.H * s.D);
      double gbs = bytes / (ms * 1e-3) / 1e9;
      printf("    kernel %d: %8.3f ms  %7.1f GB/s (%5.1f%% of peak)  [%s]\n",
             id, ms, gbs, 100.0 * gbs / peak, r.pass ? "PASS" : "FAIL");
    }
  }

  printf("\n%s\n", all_pass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
  CUDA_CHECK(cudaFree(dq));
  CUDA_CHECK(cudaFree(dK));
  CUDA_CHECK(cudaFree(dV));
  CUDA_CHECK(cudaFree(dout));
  free(hq); free(hK); free(hV); free(ho); free(href);
  return all_pass ? 0 : 1;
}

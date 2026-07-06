# Inference Kernels CUDA

Step-by-step optimized CUDA kernels for the core operations of LLM inference,
written for and measured on an **NVIDIA RTX 3060 12GB** (Ampere GA106, sm_86).

Five operations, each as a numbered progression from a naive kernel to an
optimized one:

| Op | Kernels | What it is in inference |
|---|---|---|
| GEMV | 5 | every weight matrix times the activation vector, per decoded token |
| Softmax | 4 | attention scores and output logits |
| RMSNorm | 3 | the Llama-family layer norm |
| SiLU | 3 | the MLP activation |
| Attention decode | 3 (+1 merge) | one query token attending to the whole KV cache |

Every kernel is verified against a double-precision CPU reference on multiple
shapes, including edge cases (size 1, sizes not divisible by 4, non-power-of-2
sizes). Each runner exits nonzero if any check fails. All 161 checks pass on
the hardware below.

A note on "100%": floating point kernels are never bit-identical to a
reference; fp32 summation order changes the last bits of the result. Correct
here means every output element matches the fp64 reference within a stated
tolerance (see each runner; observed errors are typically 100 to 10000 times
smaller than the tolerance). That is the same standard PyTorch and cuBLAS are
held to.

## The hardware (real numbers, printed by `./bin/device_info`)

Measured on this exact card, not copied from a spec sheet:

```
Device 0: NVIDIA GeForce RTX 3060
  Compute capability:        sm_86
  SMs:                       28
  FP32 cores total:          3584 (128 per SM)
  Boost clock:               1777 MHz
  Peak FP32 (FMA):           12737.5 GFLOPS = 12.74 TFLOPS
  Memory:                    15.0 Gbps effective, 192-bit bus
  Peak DRAM bandwidth:       360.0 GB/s
  VRAM:                      11.8 GiB
  L2 cache:                  2304 KiB
  Shared memory per SM:      100 KiB (48 KiB per block, 99 KiB with opt-in)
  Registers per SM:          65536
  Max threads per SM:        1536
```

## Why these kernels chase bandwidth, not FLOPS

The roofline crossover of this card is 12737.5 / 360 = 35.4 FLOP/byte.
Single-batch inference ops sit at about 0.5 FLOP/byte (GEMV does 2 FLOPs per
4-byte weight it reads exactly once). So the compute units can never be the
bottleneck: **the best possible fp32 GEMV on an RTX 3060 is ~180 GFLOPS, or
1.4% of the card's 12.74 TFLOPS**, and it is reached by saturating the
360 GB/s memory bus. Every optimization below is a memory optimization. This
is the opposite regime from large square GEMM, whose arithmetic intensity
grows with size (N/6 FLOP/byte for N x N fp32, i.e. ~680 at N=4096) and which
therefore chases the compute roofline with tiling; at inference decode time
there is nothing to tile because every weight is used once.

## Build and run

```bash
make            # builds bin/{device_info,gemv,softmax,rmsnorm,silu,attention}
make run        # runs everything: correctness + benchmarks
./bin/gemv 0    # one op, all kernels (0 or no argument)
./bin/gemv 3    # one op, one kernel
```

Requires CUDA 12.x. Built with `-O3 -arch=sm_86`, no `--use_fast_math` (it
would trade exp/div accuracy for speed that a memory-bound kernel cannot use).

## Measured results

All numbers below are from one `make run` on the card above (driver 535,
CUDA 12.3). Expect a few percent run-to-run variation from GPU boost clocks,
especially with a desktop session on the same GPU. Percentages are of the
360.0 GB/s theoretical peak; ~92% is the practical DRAM ceiling of GDDR6.

### 1. GEMV `y[M] = A[M,K] x[K]`

| # | Kernel | M=4096 K=4096 | M=11008 K=4096 | M=32000 K=4096 |
|---|--------|--------------|----------------|----------------|
| 1 | naive: one thread per row | 106.8 GB/s (30%) | 118.6 GB/s (33%) | 72.6 GB/s (20%) |
| 2 | one warp per row + shuffle reduction | 303.3 GB/s (84%) | 331.0 GB/s (92%) | 332.3 GB/s (92%) |
| 3 | + float4 vectorized loads | 308.6 GB/s (86%) | 321.7 GB/s (89%) | 324.7 GB/s (90%) |
| 4 | one block per row + block reduction | 301.7 GB/s (84%) | 326.2 GB/s (91%) | 320.1 GB/s (89%) |
| 5 | + ldg, unroll 4, launch_bounds | 292.7 GB/s (81%) | 331.9 GB/s (92%) | 297.4 GB/s (83%) |

The whole game is kernel 1 to kernel 2: coalescing. In kernel 1 each thread
walks its own row, so a warp touches 32 rows at once and every 32-byte memory
sector delivers 4 useful bytes. In kernel 2 the 32 lanes of a warp read 32
consecutive elements of one row: 3x faster from that change alone. Kernels
3 to 5 trade the last few percent back and forth because the bus is already
saturated; this mirrors the diminishing-returns tail of every optimization
series. The M=32000 shape is the vocabulary projection of a Llama-7B, at 1.6 ms
per token for kernel 2+.

### 2. Row-wise safe softmax on [M, N]

| # | Kernel | M=4096 N=4096 | M=32 N=32000 (logit rows) |
|---|--------|---------------|---------------------------|
| 1 | naive: one thread per row, 3 passes | 33.6 GB/s (9%) | 0.9 GB/s (0.2%) |
| 2 | one warp per row, shuffle reductions | 159.8 GB/s (44%) | 37.6 GB/s (10%) |
| 3 | one block per row + float4 | 171.0 GB/s (48%) | 144.3 GB/s (40%) |
| 4 | online softmax (fused max+sum), 2 passes | 212.1 GB/s (59%) | 185.3 GB/s (52%) |

Reported GB/s uses effective bytes (one read + one write per element), so the
3-pass kernels are penalized for their extra reads: exactly the point. Kernel
4 uses the online softmax recurrence (Milakov and Gimelshein 2018), merging
running (max, sum) pairs with `s = s1*exp(m1-M) + s2*exp(m2-M)`; one pass
computes both statistics, cutting row reads from 3 to 2 and giving the
expected ~4/3 speedup over kernel 3. The same recurrence is what makes
FlashAttention and kernel 3 of the attention op below possible. The
M=32 case shows why per-row parallelism matters: 32 rows can only occupy 32
threads in kernel 1 (0.2% of the GPU) but 32 blocks in kernels 3 and 4, a
217x end-to-end speedup.

### 3. RMSNorm on [M, N] with weight w[N]

| # | Kernel | M=4096 N=4096 | M=1 N=4096 (single decode token) |
|---|--------|---------------|-----------------------------------|
| 1 | naive: one thread per row | 44.7 GB/s (12%) | 0.206 ms |
| 2 | one block per row + block reduction | 191.2 GB/s (53%) | 0.007 ms |
| 3 | + float4 | 211.0 GB/s (59%) | 0.003 ms |

The M=1 column is time, not bandwidth: a single 4096-element row cannot fill
a 360 GB/s bus, so what matters at decode time is latency, and the optimized
kernel is 69x faster. The ~60% ceiling on the big shape comes from the norm
being two dependent passes (reduce, then scale) over rows that exceed what
registers can hold, plus per-row reduction latency; a fused
persistent-row kernel is the next step beyond this repo.

### 4. SiLU elementwise, n = 67,108,864

| # | Kernel | Bandwidth |
|---|--------|-----------|
| 1 | naive: one thread per element | 303.2 GB/s (84%) |
| 2 | grid-stride loop | 299.2 GB/s (83%) |
| 3 | grid-stride + float4 | 281.1 GB/s (78%) |

The honest lesson: a coalesced elementwise kernel is already optimal, and
"optimizations" can lose. All three sit at the practical DRAM ceiling within
noise, and float4 measures slightly slower here because wide accesses buy
nothing when the scalar kernel already issues maximal DRAM traffic. (An
earlier version launched kernel 3 with a small fixed grid, making each
thread stride 8 MB between iterations; that cost 20% in DRAM locality. The
launch now covers the data directly.)

### 5. Attention decode: out[h] = softmax(q K^T / sqrt(D)) V, H=32 heads, D=128

| # | Kernel | L=4096 | L=16384 |
|---|--------|--------|---------|
| 1 | naive: scores materialized in shared memory | 149.7 GB/s (42%) | cannot run (smem limit) |
| 2 | flash style: online softmax over tiles | 166.3 GB/s (46%) | 180.2 GB/s (50%) |
| 3 | flash-decoding: split-KV + merge kernel | **302.5 GB/s (84%)** | **317.0 GB/s (88%)** |

Kernel 1 stores all L scores in shared memory (three phases: scores, softmax,
weighted V sum), which caps L at about 12k and dies at 48 KiB of smem. Kernel
2 processes the KV cache in tiles of 128 positions with a running rescaled
accumulator, so the scores never exist in full: unbounded L, less smem, per-warp
float4 dot products. But both launch only H=32 blocks for 28 SMs, so most of
the GPU idles. Kernel 3 is flash-decoding (Dao et al. 2023): the sequence is
additionally split 8 ways, each of the 256 blocks writes an unnormalized
partial (m, s, acc[D]), and a tiny second kernel merges them with the online
softmax rule. Occupancy is restored and bandwidth doubles: 2.0x faster than
kernel 1 at L=4096, 0.44 ms per decoded token, at 84% of the theoretical peak.

## Repository layout

```
Makefile
src/
  common.cuh            CUDA_CHECK, warp/block reductions, RNG, verification, timing
  device_info.cu        prints the real numbers of the installed GPU
  kernels/
    gemv.cuh            kernels 1..5
    softmax.cuh         kernels 1..4 + online softmax merge helpers
    rmsnorm.cuh         kernels 1..3
    silu.cuh            kernels 1..3
    attention.cuh       kernels 1..3 + split-KV merge kernel
  run_gemv.cu           correctness (vs fp64 CPU) + benchmark runner
  run_softmax.cu
  run_rmsnorm.cu
  run_silu.cu
  run_attention.cu
```

## Correctness methodology

- Every kernel output is compared element-wise against a CPU reference that
  accumulates in fp64: pass iff `|got - ref| <= atol + rtol * |ref|`.
- Tolerances per op are printed by each runner (e.g. GEMV atol 5e-3 for
  values of magnitude ~64; softmax atol 1e-5 for values in [0, 1]).
- Output buffers are poisoned with NaN patterns (`cudaMemset 0xFF`) before
  each correctness launch, so a kernel that writes nothing cannot pass.
- Edge shapes are tested deliberately: size 1, dimensions not divisible by 4
  (exercising every vectorized kernel's scalar fallback and tail path),
  non-power-of-2 and prime-ish sizes, and rows/columns smaller than a block.
- Benchmark shapes are re-verified once per kernel before timing.
- Timing uses CUDA events, warmup launches, and an iteration count auto-sized
  to a ~250 ms measurement window.

## What to read next

- The FP16/INT8 versions of these kernels halve or quarter the bytes per
  weight, which on a 360 GB/s card is the only way to go materially faster
  than what is measured here. That is why quantization dominates local
  inference on this class of GPU.
- Prefill (many tokens at once) turns GEMV back into GEMM and becomes compute
  bound: that regime is won with shared-memory tiling, register blocking, and
  warptiling, a completely different optimization ladder.

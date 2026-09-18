# AGENT.md — Vidur / NeuSight Calibration

Read this before touching `neusight/Prediction/predictor.py`, any `bf16_mm*` dataset,
or any comparison table under `scripts/asplos/`. It exists because this calibration
work has already produced two false "fixes" from skipping the checks below. Follow
the checks; don't re-derive them from scratch.

## 0. The one rule that would have prevented every mistake so far

**Never trust a number labelled "measured" or "predicted" without checking its
provenance: which predictor commit, which opgraph bundle, which measurement run.**
Every incident in §4 was a provenance failure, not a maths failure. Every table this
repo produces MUST carry `predictor_version` (git SHA), `getitem_mode` or equivalent
predictor-variant tag, `measure_run_id`, `measure_source`, `n_repeats`, `measured_ci95_ms`.
A row without these is not evidence.

## 1. What Vidur/NeuSight is for here, and what it isn't

- **Purpose:** predict per-request TTFT/TPOT and per-op kernel time for Qwen2.5 /
  Mistral on A100 (and eventually cross-OEM hardware), to feed pricing and capacity
  planning — not to replace end-to-end measurement.
- **NeuSight** predicts kernel/op execution time from a traced opgraph + per-op
  predictors (wave-scaling-like for kernel-alike ops, trained MLPs for kernel-varying
  ops: Linear, BMM, LSTM/conv analogues).
- **Vidur** (separately) is the request-level scheduler simulator (continuous
  batching, KV cache, preemption) — it consumes op/kernel timings as an input, it
  does not itself know GEMM shapes.
- **It does not currently model:** FlashAttention/FlashInfer fused kernels (see §5),
  ZeRO-3 communication, gradient checkpointing recompute, PEFT/LoRA overhead, or GPU
  idle/launch-gap time between kernels. All of these are legitimate `not_modelled`
  buckets, not predictor error — see §3.

## 2. Ground rules for any comparison table

1. **Never call something "measured" unless it is a real timing.** A scaled,
   extrapolated, or formula-derived number is not measured. Column names must say so:
   `measured_ms` = real, `predicted_ms` = model output, never blend the two under one
   name. (We shipped an APE table where 17/20 "measured" values were extrapolations —
   every conclusion in that table was void.)
2. **`aicb` vs `vidur` are simulator backends, not measurement sources.** If a
   directory path contains `.../aicb/.../request_metrics.csv`, do not call it
   "measured" unless you have independently confirmed it's a real vLLM run written in
   Vidur's schema (check for non-zero per-request variance; a fixed-shape simulator
   run has near-zero CI at every concurrency — real runs don't).
3. **Use the model's own tokenizer for content-token counts, never a third-party one**
   (e.g. `cl100k_base` for a Qwen run). Token counts across tokenizers can differ by
   >10× depending on whether "one example" means one turn or one full conversation —
   define both explicitly before reporting a total.
4. **Minimum sample size before trusting a CI.** `measured_ci95_ms = 0.000` is not
   zero variance — it's too few samples for the CI method (seen at n=concurrency,
   i.e. one wave of requests at N=4 or N=8). Use the trace-replay harness with
   `min_samples_per_level >= 256`, not one wave per concurrency level.
5. **Per-module CUDA-event hook sums are NOT the measured baseline.** Hooking every
   submodule adds sync overhead; summed hook times can exceed the true end-to-end
   time by ~20% (we saw `decoder_layer_total × n_layers` = 59 ms vs a true whole-model
   forward of 48 ms). The measured source of truth is **kernel-level profiler device
   time** (`torch.profiler`, `cat == "kernel"` events), reconciled against wall time,
   not module hooks.
6. **Report signed error, not just |APE|.** "Best APE per model" hides sign and
   direction; a model that's 45% low at N=4 and 25% high at N=128 is not summarized
   by any single "best" number.

## 3. Reading an error number correctly

Before treating any gap between predicted and measured as "predictor is wrong," rule
out, in this order:
1. **Provenance** (§0/§2) — wrong bundle, stale file, wrong predictor commit.
2. **Not-modelled features** — chunked prefill, prefix caching, ZeRO-3 comms,
   gradient checkpointing, LoRA overhead, GPU idle/launch gaps between kernels
   (especially significant at low batch — bs=1 forward passes are often
   launch-bound, not compute-bound; this idle time is real but is not something a
   kernel-time predictor is trying to capture).
3. **Op-classification bugs** — an op predicted under the wrong category entirely
   (see `getitem` in §4.2: real cost ~0.5–0.8% of layer time, predicted cost was
   inflated ~2.4 ms/layer under `shape_indexing_overhead`).
4. **Genuine per-op calibration error** — a trained predictor (Linear, BMM) whose
   MAPE against real holdout data is the number you actually report.

Only #4 is "the predictor needs retraining." Don't retrain against #1–#3.

## 4. Incidents — what went wrong, so it isn't repeated

### 4.1 fp32-vs-bf16 anchoring (the original 9–11× miss)
Early Qwen2.5-7B predictions were ~9–11× too slow. Root cause: the Linear predictor
was anchored to fp32 CUDA-core throughput (~19.5 TFLOP/s, A100) while the real kernels
were `ampere_bf16_s16816gemm_*` tensor-core kernels (~170+ TFLOP/s achieved). Compute
achieved-throughput from FLOPs/time and compare to the *dtype-correct* roofline before
concluding "predictor is wrong" — a factor landing near a peak-FLOPS ratio for the
wrong dtype is a calibration-table bug, not a modelling bug.

### 4.2 `getitem` / shape_indexing_overhead over-count
NeuSight was attributing ~2 ms/layer to tensor-indexing ops (rotary embedding
slicing) that cost microseconds in reality (profiled: 0.47–0.82% of layer time, not
the ~13% implied by the raw prediction). This was diagnosed against the
**prebuilt eval opgraph** (`qwen_prebuilt_eval`) — see 4.3, it does not reproduce
against a different opgraph bundle.

### 4.3 A fix validated against one bundle silently stopped applying under another
The getitem cap was implemented and validated against `qwen_prebuilt_eval`'s
opgraph, where the phantom cost was real (~55 ms/model). When re-run against the
`qwen_bf16_mm_v2` bundle, the same predictor code was a near-no-op (the bundle's
getitem prediction was already ~0.5 ms) — so "capping" it did nothing, and one
shape's before/after numbers even *increased* slightly, which is impossible for a
`min(cur, cap)` operation and indicates either a stale/copied-forward value or a
real non-determinism bug that was never fully triaged.
**Lesson: a fix is scoped to the exact bundle+predictor-commit it was validated
against. Re-validate on every bundle change. If a `min`/`max` cap appears to move a
value the wrong direction, suspect stale data before suspecting the arithmetic.**

### 4.4 Proxy architecture mismatches inflate with model size
Qwen2.5 uses GQA (7B: 28 Q / 4 KV heads; 72B: 64/8), QKV bias, SwiGLU (3 matrices,
not 2), and a 152k vocab. A GPT-2/Llama-style proxy without these corrections
over-counts attention/KV work and the error compounds with layer count — this is
why error grew monotonically 7B → 14B → 32B → 72B (68% → 533% in one run) even
though the underlying predictor bug was constant. Implied-throughput sanity check
(GFLOP / predicted_ms) should be roughly equal across model sizes if the proxy is
correct; if 14B's implied throughput is half of 7B/32B/72B's, the 14B proxy config
(`intermediate_size`, `num_key_value_heads`) is wrong — diff it against the real HF
`config.json` before touching the predictor.

### 4.5 FA2 vs eager are structurally different execution paths
FlashAttention-2 replaces `aten::bmm` + `aten::_softmax` with one fused kernel
(`flash_attn::_flash_attn_forward`). A BMM predictor trained/validated on eager
traces has **no coverage** of the FA2 path — confirmed by op-presence diff (eager:
bmm+softmax present, flash op absent; FA2: reverse). Never populate `predicted_ms`
for FA2 rows using an eager-trained BMM predictor; mark `not_available` until a
flash-kernel model exists (roofline + fitted efficiency η, fit per (batch, seqlen)
bucket from ~15–20 real flash-kernel timings — this is a half-day task, not a new
dataset).

Measured FA2 vs eager divergence (n=20-30, real A100):
| Shape | Eager | FA2 | Δ |
|---|---|---|---|
| (512,1) | ~48.6 ms | ~48.2 ms | ~−1% (near-tie; attention is negligible at this shape) |
| (2048,1) | ~241.3 ms | ~150.9 ms | ~−37% (1.6× speedup) |
| (512,8) | ~324.5 ms | ~283.5 ms | ~−13% (1.14× speedup) |

Divergence grows with seq² and batch, as expected from eager's materialized
O(s²) attention matrix. **Don't validate an attention predictor at (512,1) only —
that shape can't distinguish "backend doesn't matter" from "predictor has no
attention term at all."**

### 4.6 Small-M GEMM efficiency is likely under-modelled
Cross-checking FA2 timings: (2048,1) is only ~3.1× the cost of (512,1) for 4× the
tokens, once attention is ~linear — implying cuBLAS/tensor-core efficiency rises
materially from M=512 to M=2048. A LINEAR predictor trained mostly on large-M
shapes will fit large-M well and silently under-predict at M≈512, exactly the
shape most fine-tuning batches run at. **Report holdout MAPE binned by M
(≤512, 512–2048, >2048), never just the aggregate** — the aggregate can hide a bad
small-M fit behind good large-M fits.

## 5. Standing constraints (do not re-derive, do not violate)

- **Gate any final accuracy claim on the FA2 path, not eager.** Eager is a useful
  diagnostic (it exposes op-level bugs cleanly because ops don't get fused) but
  production serving is vLLM+FlashAttention. An eager-only pass is not sufficient
  evidence of production readiness.
- **BMM retrain triage rule:** skip retraining the BMM predictor unless the
  reconciliation table (see below) shows measured-vs-predicted BMM residual >5 ms on
  an eager row. It's structurally irrelevant to FA2 (§4.5) and not worth the effort
  unless eager itself needs it for some other reason.
- **Before any retrain, build the kernel-level reconciliation table:**
  `measured (profiler device-time) vs predicted`, broken out by category — GEMM
  (`aten::mm`/`addmm`), BMM, elementwise/copy, norm/reduce, softmax, and an explicit
  **idle/launch-gap** row (wall time − Σ kernel device time). Know which category is
  wrong, and by how much and in which direction, before changing training data.
- **Any retrain must report M-binned holdout MAPE**, and the small-M bin is the
  decision signal for whether 7B (and other models at production batch sizes)
  will actually improve — not the aggregate MAPE.
- **Pre-register the expected outcome before running a fix.** Write down, in this
  file or the run's notes, what ratio/error you expect the fix to produce and why
  (back-of-envelope from FLOPs/roofline). If the result lands materially outside that
  band, that's a signal of a second (or third) uncorrected term — treat it as a new
  finding, log it here, don't just accept whatever number comes out.
- **Current status (update this section on every material change — do not let it go
  stale):**
  - Bundle in use: `qwen_bf16_mm_v2`. Predictor commit: track via sha256 of
    `predictor.py`, log alongside git SHA (working tree may be dirty — note that too).
  - Eager 7B baseline (this bundle, absolute-cap getitem mode): predicted/measured
    ratios ≈ **1.62× (512,1), 1.33× (2048,1), 1.48× (512,8)** — net over-prediction,
    cause not yet isolated. The earlier "getitem sign-flip to under-prediction"
    result (§4.3) does NOT apply to this bundle — do not reuse those numbers or the
    0.24–0.9× expected band derived from them.
  - Open/untriaged: a `(512,1)` predicted value changed between two supposedly
    equivalent predictor runs in a way a `min(cur, cap)` operation cannot produce.
    Rerun (512,1) twice per mode in one sitting and diff before trusting that row in
    any table.
  - Next step in the agreed order: kernel-level reconciliation table → small-M
    LINEAR retrain (data already collected: `small_m_linear_measurements.csv`, 102
    shapes, M ∈ {128,256,384,512,768,1024} × 17 Qwen-derived (N,K) pairs) → re-run
    eager diagnostic → FA2 flash-kernel model → gate final claim on FA2 rows with
    M-binned MAPE + pre-registered band + pass/fail/third-term-trigger table.

## 6. Definitions (stop re-litigating these)

- **Content tokens** vs **logged/padded tokens**: throughput and cost figures use
  content tokens (post-truncation, post-masking, model's own tokenizer). Padding is
  recorded separately and never used in a headline metric.
- **A "measured" row** has a real device timing behind it, with `n_repeats`,
  `measured_ci95_ms`, warm-up steps excluded, and a `measure_run_id` you could go
  re-open. Everything else is `predicted`, `extrapolated`, or `not_available` —
  pick the honest one.

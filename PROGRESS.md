# DiT KV-Routing: Stale-K as a Unified Attention Index

## Paper thesis
"K is a stable routing index, V is volatile payload." In video DiTs, attention
scores computed from a stale (low-bit) K snapshot remain accurate enough for
token selection across denoising steps, while V must be recomputed.
A single stale int4 K cache can therefore serve THREE roles at once:
  1. routing index for top-k sparse attention
  2. importance score for KV eviction (AR / streaming)
  3. quantized storage of the index itself
No prior work unifies these: TempoKV (arXiv 2610.15324) quantizes K/V but does
no sparsity; DeCoPrune evicts but does not quantize; ~30 sparse-attention
papers (WaveAlign, AB-Attention, LoSA, ...) all compute masks from FRESH K.

## Experiment 1: DiT-XL/2-256 offline probe (DONE)
- K-only reuse error (gap=1): 0.026 ~ feature-cache error 0.028
- V reuse error: 0.23 (bad); K/V are 4-bit friendly, features are NOT
- => asymmetry confirmed at image-DiT scale

## Experiment 2: DiT-XL end-to-end (DONE, results/dit_xl/e2e_metrics.json)
64 paired images, PSNR/SSIM vs exact:
- K-only stride2: PSNR 28.8 / SSIM 0.981  (beats feat2 25.6/0.962, kv2 25.3/0.958)
- k4: 22.6, k8: 18.6, int4-K(kq4): 22.4 ~= k4, feat4: 20.5

## Experiment 3: CogVideoX-2b probe (DONE, results/cogvideox/analysis.json)
17.9k joint-attention tokens, layers {8,22}, steps {5..45}, gaps {1,2,5,10}:
- stale int4-K top-k routing ~= oracle fresh-K routing:
  coverage 97% @ 50%, 92% @ 25%; mask IoU 0.73-0.83
- gap-immune: coverage drops only ~2% from gap=1 to gap=10
- masked recompute error 0.03 @ 50%
- K drift 0.11/step vs V drift 0.06-0.09 (post-RoPE)

## Experiment 4: CogVideoX-2b END-TO-END (IN PROGRESS)
Stale int4-K routed top-k sparse attention inside a real 50-step generation.
Runs: oracle50 / stale50 / stale25 / stale10 vs dense base.
Metric: PSNR/SSIM per frame vs base. Script: code/cogvid_e2e3.py
Status: running on GPUs 1-4 (2026-11-03 ~07:30 CST).

## Related work triage (2026-11-03)
- StreamDiT (2507.03745): training-side streaming T2V, closed-source, not comparable
- TempoKV: predictive KV quant for AR video; K-predictable/V-history asymmetry = our result
- CORAL-KV, DeCoPrune, FlashForward, Memory Forcing: each covers ONE mechanism
- UnStep: training-free causal-model accel; orthogonal to our index idea
- GAP: unified index layer for bidirectional video DiTs, streaming-ification
  of open models (CogVideoX / Wan2.1) remains open

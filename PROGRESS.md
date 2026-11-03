# Stale-K Routing: K as Index, V as Payload — Progress Log

## Thesis
In DiT/video-DiT attention, **K is a stable routing index, V is volatile payload**.
Cache INT4-quantized stale K across diffusion steps to route sparse attention;
compute only selected keys/values fresh. Unifies cache × quant × sparse —
a combination the literature treats as mutually exclusive.

## Verified empirical findings

### DiT-XL/2 256x256 (offline probe + e2e, 64 paired imgs)
- K-only reuse error (gap=1): 0.026 vs feature cache 0.028 — comparable.
- V reuse ≈ attn_out reuse: ~0.23 error — bad.
- K/V quant-friendly at 4bit; feature quant at 4bit catastrophic (0.24+).
- e2e PSNR vs exact (50 steps, seed-paired):
  - k2 (K-only, stride2): 28.8 / SSIM 0.981  ← beats feat2 (25.6) and kv2 (25.3)
  - k4: 22.6 | k8: 18.6 | kq4 (INT4): 22.4 ≈ k4 | feat4: 20.5

### CogVideoX-2b (17.9k joint tokens, post-RoPE capture)
- K drift 0.11/step vs V drift 0.06-0.09 (asymmetry persists post-RoPE).
- Stale INT4-K top-k routing ≈ oracle (fresh K):
  - coverage 97% @ r=0.5, 92% @ r=0.25
  - mask IoU 0.73-0.83; near gap-immunity (coverage -2% gap1->10)
  - masked recompute err 0.03 @ r=0.5

### CogVideoX e2e (IN PROGRESS — /root/cogvid_e2e.py)
Running: base | oracle r=0.5 | stale-int4 r=0.5 | stale-int4 r=0.25
Same seed (generator seed=0), 50 steps, 49 frames, CogVideoX-2b.
Metric: PSNR/SSIM vs base video.

## Landscape check (arXiv, Nov 2026)
- AR video KV cache + dLLM KV cache: RED OCEAN.
- Bidirectional DiT K/V-level cross-step reuse: TRUE GAP — TeaCache/FORA
  cache block features, never K/V.
- RECAP (arXiv 2610.26244) couples sparse+cache for AR models — closest
  competitor, cite it.

## Next steps
- [ ] Finish CogVideoX e2e table (this run)
- [ ] Wan2.1-T2V-1.3B replication (smaller, RoPE variant)
- [ ] Speedup measurement with real sparse kernel (flex_attn/block mask)
- [ ] Multi-prompt FID/LPIPS sweep

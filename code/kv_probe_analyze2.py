"""Gap-resolved reuse error: KV-cache vs feature-cache vs quantized-KV."""
import os, json, sys
import numpy as np
import torch

CAP = "/root/kv_probe/captures"
OUT = "/root/kv_probe/results"
device = "cpu"
NL, H, D = 28, 16, 72

def load_step(i):
    return torch.load(os.path.join(CAP, f"step_{i:03d}", "tensors.pt"),
                      map_location="cpu", weights_only=False)

meta = json.load(open(os.path.join(CAP, "meta.json")))
S = len([d for d in os.listdir(CAP) if d.startswith("step_")])

from diffusers import DiTPipeline
pipe = DiTPipeline.from_pretrained(meta["model"], torch_dtype=torch.float32)
blocks = list(pipe.transformer.transformer_blocks)
wout = {i: (blocks[i].attn1.to_out[0].weight.detach(),
            blocks[i].attn1.to_out[0].bias.detach()) for i in range(NL)}
del pipe

def heads(x):
    B, N, C = x.shape
    return x.view(B, N, H, D).transpose(1, 2)

def attn_ctx(q, k, v):
    p = torch.softmax(heads(q) @ heads(k).transpose(-1, -2) / (D ** 0.5), dim=-1)
    ctx = (p @ heads(v)).transpose(1, 2)
    return ctx.reshape(ctx.shape[0], ctx.shape[1], -1)

def proj(ctx, li):
    w, b = wout[li]
    return ctx @ w.T + b

def fq(x, bits):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    s = amax / (2 ** (bits - 1) - 1)
    return (x / s).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * s

def rl2(a, b):
    return ((a - b).norm() / (b.norm() + 1e-12)).item()

gaps = [1, 2, 4, 8, 16]
metrics = ["kv", "kv4", "kv8", "k_only", "v_only", "feat_attn", "feat_block", "feat_block4"]
res = {m: np.zeros((len(gaps), NL)) for m in metrics}
cnt = np.zeros((len(gaps), NL))

steps = list(range(0, S, 2))
_cache = {}
def get(t):
    if t not in _cache:
        _cache[t] = load_step(t)
    return _cache[t]
print("lazy loading")

for t in steps:
    cur = get(t)
    for gi, g in enumerate(gaps):
        src_t = t - g
        if src_t < 0:
            continue
        src = get(src_t)
        for li in range(NL):
            q_ = cur[f"{li:02d}_q"].float()
            kf, vf = cur[f"{li:02d}_k"].float(), cur[f"{li:02d}_v"].float()
            ks, vs = src[f"{li:02d}_k"].float(), src[f"{li:02d}_v"].float()
            ref = cur[f"{li:02d}_attn_out"].float()
            refb = cur[f"{li:02d}_block_out"].float()
            res["kv"][gi, li] += rl2(proj(attn_ctx(q_, ks, vs), li), ref)
            res["kv4"][gi, li] += rl2(proj(attn_ctx(q_, fq(ks, 4), fq(vs, 4)), li), ref)
            res["kv8"][gi, li] += rl2(proj(attn_ctx(q_, fq(ks, 8), fq(vs, 8)), li), ref)
            res["k_only"][gi, li] += rl2(proj(attn_ctx(q_, ks, vf), li), ref)
            res["v_only"][gi, li] += rl2(proj(attn_ctx(q_, kf, vs), li), ref)
            res["feat_attn"][gi, li] += rl2(src[f"{li:02d}_attn_out"].float(), ref)
            res["feat_block"][gi, li] += rl2(src[f"{li:02d}_block_out"].float(), refb)
            res["feat_block4"][gi, li] += rl2(fq(src[f"{li:02d}_block_out"].float(), 4), refb)
            cnt[gi, li] += 1
    if t % 10 == 0:
        print("step", t, flush=True)

for m in metrics:
    res[m] /= np.maximum(cnt, 1)
np.savez(os.path.join(OUT, "reuse_gap.npz"), gaps=gaps, **res)
print(json.dumps({m: res[m].mean(axis=1).tolist() for m in metrics}, indent=0))

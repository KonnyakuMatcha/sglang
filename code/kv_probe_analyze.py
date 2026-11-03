"""Offline analysis of captured DiT activations: drift, quantization sensitivity, reuse error."""
import os, json, sys
import numpy as np
import torch

CAP = sys.argv[1] if len(sys.argv) > 1 else "/root/kv_probe/captures"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/kv_probe/results"
os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
NL, H, D = 28, 16, 72   # layers, heads, head_dim for DiT-XL

def load_step(i):
    return torch.load(os.path.join(CAP, f"step_{i:03d}", "tensors.pt"),
                      map_location="cpu", weights_only=False)

meta = json.load(open(os.path.join(CAP, "meta.json")))
S = len([d for d in os.listdir(CAP) if d.startswith("step_")])
print(f"steps={S}")

def rel_l2(a, b):
    return ((a - b).norm() / (b.norm() + 1e-12)).item()

def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0).item()

# ---------------- 1. drift curves ----------------
print("=== drift analysis ===", flush=True)
drift = {k: np.zeros((NL, S - 1)) for k in ["q", "k", "v", "attn_out", "block_out"]}
drift_cos = {k: np.zeros((NL, S - 1)) for k in drift}
prev = load_step(0)
for t in range(1, S):
    cur = load_step(t)
    for li in range(NL):
        for k in drift:
            a = cur[f"{li:02d}_{k}"].float()
            b = prev[f"{li:02d}_{k}"].float()
            drift[k][li, t - 1] = rel_l2(a, b)
            drift_cos[k][li, t - 1] = cos_sim(a, b)
    prev = cur
    if t % 10 == 0:
        print(f"  step {t}", flush=True)
np.savez(os.path.join(OUT, "drift.npz"), **{f"{k}_l2": v for k, v in drift.items()},
         **{f"{k}_cos": v for k, v in drift_cos.items()})

# ---------------- 2-4 need attention recompute ----------------
print("=== loading transformer for W_out ===", flush=True)
from diffusers import DiTPipeline
pipe = DiTPipeline.from_pretrained(meta["model"], torch_dtype=torch.float32)
blocks = list(pipe.transformer.transformer_blocks)
wout = {i: (blocks[i].attn1.to_out[0].weight.detach(),
            blocks[i].attn1.to_out[0].bias.detach()) for i in range(NL)}
del pipe
torch.cuda.empty_cache() if device == "cuda" else None

def heads(x):  # (B,N,C) -> (B,H,N,D)
    B, N, C = x.shape
    return x.view(B, N, H, D).transpose(1, 2)

def attn_ctx(q, k, v):
    p = torch.softmax(heads(q) @ heads(k).transpose(-1, -2) / (D ** 0.5), dim=-1)
    ctx = (p @ heads(v)).transpose(1, 2)  # B,N,H,D
    return ctx.reshape(ctx.shape[0], ctx.shape[1], -1)

def proj(ctx, li):
    w, b = wout[li]
    return ctx @ w.T + b

def fake_quant(x, bits, group="token"):
    # symmetric absmax quant, per-token (last dim grouping = per token vector)
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / (2 ** (bits - 1) - 1)
    return (x / scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * scale

# ---------------- 2. reuse error: KV-cache vs feature-cache, at all gaps ----
print("=== reuse error ===", flush=True)
gaps = [1, 2, 4, 8]
reuse_kv = np.zeros((NL, S))       # rel-L2 of attn_out when K,V reused from t-gap
reuse_feat = np.zeros((NL, S))     # attn_out[t-gap] directly reused
reuse_kvq4 = np.zeros((NL, S))     # KV reused AND stored at 4bit
reuse_kv_only = np.zeros((NL, S))
reuse_v_only = np.zeros((NL, S))
reuse_counts = np.zeros((NL, S))
GAP = max(gaps)
cache = {}  # step -> tensors
for t in range(S):
    cur = load_step(t)
    cache[t] = {k: cur[k] for k in cur} if t < GAP else None
    for g in gaps:
        src_t = t - g
        if src_t < 0:
            continue
        src = load_step(src_t)
        for li in range(NL):
            q = cur[f"{li:02d}_q"].float()
            k_f, v_f = cur[f"{li:02d}_k"].float(), cur[f"{li:02d}_v"].float()
            k_s, v_s = src[f"{li:02d}_k"].float(), src[f"{li:02d}_v"].float()
            ref = cur[f"{li:02d}_attn_out"].float()
            # fresh-context sanity: recompute with fresh K,V
            out_kv = proj(attn_ctx(q, k_s, v_s), li)
            out_kvq = proj(attn_ctx(q, fake_quant(k_s, 4), fake_quant(v_s, 4)), li)
            out_ko = proj(attn_ctx(q, k_s, v_f), li)
            out_vo = proj(attn_ctx(q, k_f, v_s), li)
            out_feat = src[f"{li:02d}_attn_out"].float()
            reuse_kv[li, t] += rel_l2(out_kv, ref)
            reuse_kvq4[li, t] += rel_l2(out_kvq, ref)
            reuse_kv_only[li, t] += rel_l2(out_ko, ref)
            reuse_v_only[li, t] += rel_l2(out_vo, ref)
            reuse_feat[li, t] += rel_l2(out_feat, ref)
            reuse_counts[li, t] += 1
    if t % 10 == 0:
        print(f"  step {t}", flush=True)
reuse_kv /= np.maximum(reuse_counts, 1)
reuse_kvq4 /= np.maximum(reuse_counts, 1)
reuse_kv_only /= np.maximum(reuse_counts, 1)
reuse_v_only /= np.maximum(reuse_counts, 1)
reuse_feat /= np.maximum(reuse_counts, 1)
np.savez(os.path.join(OUT, "reuse.npz"),
         kv=reuse_kv, kvq4=reuse_kvq4, k_only=reuse_kv_only,
         v_only=reuse_v_only, feat=reuse_feat, counts=reuse_counts)

# ---------------- 3. fresh quantization sensitivity ----------------
print("=== quantization sensitivity ===", flush=True)
bits_list = [8, 6, 4, 3]
qsens = {m: np.zeros((NL, S)) for m in
         ["k", "v", "kv", "attn_direct"]}   # quant K only / V only / both / attn_out directly
qsens_cnt = np.zeros((NL, S))
for t in range(0, S, 2):   # every other step to save time
    cur = load_step(t)
    for li in range(NL):
        q = cur[f"{li:02d}_q"].float()
        k = cur[f"{li:02d}_k"].float()
        v = cur[f"{li:02d}_v"].float()
        ref = cur[f"{li:02d}_attn_out"].float()
        ctx_ref = attn_ctx(q, k, v)
        for b in bits_list:
            pass
        # record per-bit in a dict
        for b in bits_list:
            key = lambda m: qsens[m][li, t]
            qsens.setdefault(f"perbit", None)
        # accumulate per-bit separately below
        for b in bits_list:
            ok = proj(attn_ctx(q, fake_quant(k, b), v), li)
            ov = proj(attn_ctx(q, k, fake_quant(v, b)), li)
            okv = proj(attn_ctx(q, fake_quant(k, b), fake_quant(v, b)), li)
            of = fake_quant(ref, b)
            if b == 4:  # store detailed only for 4-bit headline
                qsens["k"][li, t] = rel_l2(ok, ref)
                qsens["v"][li, t] = rel_l2(ov, ref)
                qsens["kv"][li, t] = rel_l2(okv, ref)
                qsens["attn_direct"][li, t] = rel_l2(of, ref)
        qsens_cnt[li, t] = 1
    if t % 10 == 0:
        print(f"  step {t}", flush=True)
np.savez(os.path.join(OUT, "qsens.npz"), **qsens)

# also per-bit sweep at a few steps for curve
bit_sweep = np.zeros((len(bits_list), NL))
steps_probe = [S // 2]
for t in steps_probe:
    cur = load_step(t)
    for li in range(NL):
        q = cur[f"{li:02d}_q"].float(); k = cur[f"{li:02d}_k"].float(); v = cur[f"{li:02d}_v"].float()
        ref = cur[f"{li:02d}_attn_out"].float()
        for bi, b in enumerate(bits_list):
            okv = proj(attn_ctx(q, fake_quant(k, b), fake_quant(v, b)), li)
            bit_sweep[bi, li] += rel_l2(okv, ref) / len(steps_probe)
np.savez(os.path.join(OUT, "bit_sweep.npz"), bits=bits_list, err=bit_sweep)

# ---------------- 4. per-token heterogeneity ----------------
print("=== per-token drift ===", flush=True)
tok_drift_k = []   # per-token rel-L2 of K between t,t+1 at sample steps
tok_drift_b = []
for t in range(0, S - 1, 5):
    a, b = load_step(t), load_step(t + 1)
    li = NL // 2
    ka = a[f"{li:02d}_k"].float(); kb = b[f"{li:02d}_k"].float()
    ba = a[f"{li:02d}_block_out"].float(); bb = b[f"{li:02d}_block_out"].float()
    dk = (ka - kb).norm(dim=-1) / (kb.norm(dim=-1) + 1e-12)   # (B,N)
    db = (ba - bb).norm(dim=-1) / (bb.norm(dim=-1) + 1e-12)
    tok_drift_k.append(dk.flatten().numpy())
    tok_drift_b.append(db.flatten().numpy())
np.savez(os.path.join(OUT, "tokdrift.npz"),
         k=np.concatenate(tok_drift_k), block=np.concatenate(tok_drift_b))

# ---------------- summary ----------------
summ = {
    "drift_mean_rell2": {k: float(v.mean()) for k, v in drift.items()},
    "drift_cos_mean": {k: float(v.mean()) for k, v in drift_cos.items()},
    "reuse_gap_mean": {
        "kv": float(reuse_kv.mean()), "kvq4": float(reuse_kvq4.mean()),
        "k_only": float(reuse_kv_only.mean()), "v_only": float(reuse_v_only.mean()),
        "feat": float(reuse_feat.mean())},
    "qsens_4bit": {k: float(v[qsens_cnt > 0].mean()) for k, v in qsens.items()},
}
json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print(json.dumps(summ, indent=2))

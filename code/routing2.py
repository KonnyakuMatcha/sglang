"""Routing-index analysis v2: decompose stale-K routing error vs sparsity error.
For each (layer, step, gap, frac):
  mask_stale = topk(q @ kq4_old)   mask_oracle = topk(q @ k_fresh)
  cov_stale/cov_oracle on true weights; err_sparse_stale/err_sparse_oracle vs dense fresh.
  Also top-p (cumulative mass p) masks.
"""
import numpy as np, torch, os

def sel_topk(s, frac):
    k = max(1, int(s.shape[-1] * frac))
    idx = s.topk(k, dim=-1).indices
    m = torch.zeros_like(s, dtype=torch.bool); m.scatter_(-1, idx, True); return m

def sel_topp(s, p):
    sv, idx = torch.sort(s, dim=-1, descending=True)
    w = torch.softmax(sv, -1)
    cum = w.cumsum(-1)
    keep = cum <= p
    keep[..., 0] = True
    m = torch.zeros_like(s, dtype=torch.bool)
    m.scatter_(-1, idx, keep); return m

def attn_masked(q, k, v, mask, scale):
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    s = s.masked_fill(~mask, float('-inf'))
    return torch.matmul(torch.softmax(s, -1), v)

def quant(k, nbits=4):
    qmax = 2 ** (nbits - 1) - 1
    sc = k.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    return (k / sc).round().clamp(-qmax, qmax) * sc

def load(t):
    return torch.load(f'/root/kv_probe/captures/step_{t:03d}/tensors.pt', map_location='cpu', weights_only=True)

def rh(x):
    B, S, C = x.shape
    return x.float().view(B, S, 16, C // 16).transpose(1, 2)

LAYERS = [0, 5, 10, 15, 20, 27]
GAPS = [2, 4, 8]
FRACS = [0.75, 0.5, 0.25, 0.1]
TOPP = [0.9, 0.8]
STEPS = [5, 10, 20, 30, 40]
out = {}
for t in STEPS:
    cur = load(t)
    olds = {g: load(t - g) for g in GAPS if t - g >= 0}
    for li in LAYERS:
        q, kf, vf = rh(cur[f'{li:02d}_q']), rh(cur[f'{li:02d}_k']), rh(cur[f'{li:02d}_v'])
        D = q.shape[-1]; scale = 1.0 / np.sqrt(D)
        s_true = torch.matmul(q, kf.transpose(-1, -2)) * scale
        w_true = torch.softmax(s_true, -1)
        o_true = torch.matmul(w_true, vf)
        for g, old in olds.items():
            ko = quant(rh(old[f'{li:02d}_k']), 4)
            s_stale = torch.matmul(q, ko.transpose(-1, -2)) * scale
            for fr in FRACS:
                mo = sel_topk(s_true, fr)
                ms = sel_topk(s_stale, fr)
                out[f'L{li}_t{t}_g{g}_f{fr}'] = dict(
                    cov_s=(w_true * ms.float()).sum(-1).mean().item(),
                    cov_o=(w_true * mo.float()).sum(-1).mean().item(),
                    err_s=((attn_masked(q, kf, vf, ms, scale) - o_true).norm() / o_true.norm()).item(),
                    err_o=((attn_masked(q, kf, vf, mo, scale) - o_true).norm() / o_true.norm()).item(),
                    iou=(ms & mo).sum().item() / (ms | mo).sum().item())
            for p in TOPP:
                ms = sel_topp(s_stale, p)
                out[f'L{li}_t{t}_g{g}_p{p}'] = dict(
                    cov_s=(w_true * ms.float()).sum(-1).mean().item(),
                    frac=ms.float().mean().item(),
                    err_s=((attn_masked(q, kf, vf, ms, scale) - o_true).norm() / o_true.norm()).item())
        del cur[f'{li:02d}_q'], cur[f'{li:02d}_k'], cur[f'{li:02d}_v']
    del cur, olds
    print(t, flush=True)
np.savez('/root/kv_probe/results/routing2.npz',
         **{k: np.array(list(v.values())) for k, v in out.items()})
import json
json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in out.items()},
          open('/root/kv_probe/results/routing2.json', 'w'))
for fr in FRACS:
    rows = [(v['cov_s'], v['cov_o'], v['err_s'], v['err_o'], v['iou'])
            for k, v in out.items() if f'_f{fr}' in k]
    r = np.array(rows)
    print(f'topk frac={fr}: cov_stale={r[:,0].mean():.3f} cov_oracle={r[:,1].mean():.3f} '
          f'err_stale={r[:,2].mean():.3f} err_oracle={r[:,3].mean():.3f} iou={r[:,4].mean():.3f}')
for p in TOPP:
    rows = [(v['cov_s'], v['frac'], v['err_s']) for k, v in out.items() if f'_p{p}' in k]
    r = np.array(rows)
    print(f'topp p={p}: cov={r[:,0].mean():.3f} kept={r[:,1].mean():.3f} err={r[:,2].mean():.3f}')

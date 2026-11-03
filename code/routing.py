"""Routing-index analysis on captured DiT activations.
Q: can stale(4bit) K from step t-g route top-k selection whose dense recompute
   with FRESH K,V recovers the dense output?
Metrics per (layer, step, gap, k_frac):
  - coverage: ||sum of true attn weights on selected keys|| / 1
  - sparse-dense rel err: ||attn_sparse_fresh - attn_dense_fresh|| / ||attn_dense||
"""
import numpy as np, torch, glob, sys, os

def sel_topk(scores, frac):
    # scores (H,S,S) -> bool mask (H,S,S) keeping top frac keys per query
    k = max(1, int(scores.shape[-1] * frac))
    idx = scores.topk(k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask

def attn_with_mask(q, k, v, mask, scale):
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    s = s.masked_fill(~mask, float('-inf'))
    w = torch.softmax(s, dim=-1)
    return torch.matmul(w, v), w

def quant(k, nbits=4):
    qmax = 2 ** (nbits - 1) - 1
    sc = k.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    return (k / sc).round().clamp(-qmax, qmax) * sc

def load_step(d):
    return torch.load(os.path.join(d, 'tensors.pt'), map_location='cpu', weights_only=True)

LAYERS = [0, 5, 10, 15, 20, 27]
GAPS = [2, 4, 8]
FRACS = [0.5, 0.25, 0.1]
STEPS = [5, 10, 20, 30, 40]
CAP = '/root/kv_probe/captures'

def main():
    out = {}
    for t in STEPS:
        cur = load_step(f'{CAP}/step_{t:03d}')
        for g in GAPS:
            if t - g < 0: continue
            old = load_step(f'{CAP}/step_{t-g:03d}')
            for li in LAYERS:
                def rh(x):
                    B, S, C = x.shape
                    return x.float().view(B, S, 16, C // 16).transpose(1, 2)  # (B,H,S,D)
                q = rh(cur[f'{li:02d}_q'])
                kf = rh(cur[f'{li:02d}_k'])
                vf = rh(cur[f'{li:02d}_v'])
                ko = quant(rh(old[f'{li:02d}_k']), 4)
                H, S, D = q.shape[1], q.shape[2], q.shape[3]
                scale = 1.0 / np.sqrt(D)
                s_true = torch.matmul(q, kf.transpose(-1, -2)) * scale
                w_true = torch.softmax(s_true, -1)
                o_true = torch.matmul(w_true, vf)
                s_stale = torch.matmul(q, ko.transpose(-1, -2)) * scale
                for fr in FRACS:
                    mask = sel_topk(s_stale, fr)
                    cov = (w_true * mask.float()).sum(-1).mean().item()
                    o_sp, _ = attn_with_mask(q, kf, vf, mask, scale)
                    err = ((o_sp - o_true).norm() / o_true.norm()).item()
                    key = f'L{li}_t{t}_g{g}_f{fr}'
                    out[key] = dict(cov=cov, err=err)
            del old
        del cur
        print(t, flush=True)
    np.savez('/root/kv_probe/results/routing.npz', **{kk: np.array([v['cov'], v['err']]) for kk, v in out.items()})
    # print summary
    for fr in FRACS:
        for g in GAPS:
            covs = [v['cov'] for k, v in out.items() if f'_g{g}_f{fr}' in k]
            errs = [v['err'] for k, v in out.items() if f'_g{g}_f{fr}' in k]
            print(f'frac={fr} gap={g}: cov={np.mean(covs):.3f} err={np.mean(errs):.3f}')

main()

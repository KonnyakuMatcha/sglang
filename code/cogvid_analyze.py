"""CogVideoX probe analysis.
1) K vs V drift per step (L2 rel, cond row only)
2) stale-4bit-K routing coverage vs oracle at anchor steps, gaps {1,2,5,10}
3) K-only-reuse error at anchors: ||attn(q,k_old,v_fresh)-attn_dense||/||dense||
All per-head chunked on GPU.
"""
import torch, numpy as np, json, os, math

OUT = '/root/cogvid_probe'
LAYERS = [8, 22]
ANCHORS = [5, 10, 15, 20, 25, 30, 35, 40, 45]
GAPS = [1, 2, 5, 10]
FRACS = [0.5, 0.25, 0.1]
DEV = 'cuda'
DT = torch.bfloat16

def quant(k, nbits=4):
    qmax = 2 ** (nbits - 1) - 1
    sc = k.abs().amax(-1, keepdim=True).clamp_min(1e-6).float() / qmax
    return ((k.float() / sc).round().clamp(-qmax, qmax) * sc)

def rh(x, heads):
    # stored (1,H,S,D) post-head-split -> (H,S,D)
    return x.squeeze(0)

def head_scores(q, k):
    # q,k (S,D) bf16 -> (S,S) fp32
    return (q.float() @ k.float().T) / math.sqrt(q.shape[-1])

res = {'drift': [], 'route': [], 'kreuse': []}
steps = sorted(int(f[5:8]) for f in os.listdir(OUT) if f.startswith('step_'))
for i, t in enumerate(steps):
    if i == 0:
        continue
    cur = torch.load(f'{OUT}/step_{t:03d}.pt', weights_only=True, map_location='cpu')
    prev = torch.load(f'{OUT}/step_{t-1:03d}.pt', weights_only=True, map_location='cpu')
    for li in LAYERS:
        kd = ((cur[f'{li:02d}_k'].float() - prev[f'{li:02d}_k'].float()).norm()
              / prev[f'{li:02d}_k'].float().norm()).item()
        vd = None
        if f'{li:02d}_v' in cur and f'{li:02d}_v' in prev:
            vd = ((cur[f'{li:02d}_v'].float() - prev[f'{li:02d}_v'].float()).norm()
                  / prev[f'{li:02d}_v'].float().norm()).item()
        res['drift'].append(dict(t=t, li=li, k=kd, v=vd))
    del cur, prev
    print('drift', t, flush=True)

for t in ANCHORS:
    cur = torch.load(f'{OUT}/step_{t:03d}.pt', weights_only=True, map_location='cpu')
    for li in LAYERS:
        q = rh(cur[f'{li:02d}_q'], 0).to(DEV, DT)
        kf = rh(cur[f'{li:02d}_k'], 0).to(DEV, DT)
        vf = rh(cur[f'{li:02d}_v'], 0).to(DEV, DT)
        S = q.shape[1]
        for g in GAPS:
            if t - g < 0: continue
            old = torch.load(f'{OUT}/step_{t-g:03d}.pt', weights_only=True, map_location='cpu')
            ko = quant(rh(old[f'{li:02d}_k'], 0).to(DEV)).to(DT)
            del old
            for h in range(q.shape[0]):
                s_true = head_scores(q[h], kf[h])
                w_true = torch.softmax(s_true, -1)
                s_stale = head_scores(q[h], ko[h])
                for fr in FRACS:
                    kk = max(1, int(S * fr))
                    mo = torch.zeros_like(s_true, dtype=torch.bool)
                    mo.scatter_(-1, s_true.topk(kk, -1).indices, True)
                    ms = torch.zeros_like(s_true, dtype=torch.bool)
                    ms.scatter_(-1, s_stale.topk(kk, -1).indices, True)
                    cov_s = (w_true * ms).sum(-1).mean().item()
                    cov_o = (w_true * mo).sum(-1).mean().item()
                    iou = (ms & mo).sum().item() / (ms | mo).sum().item()
                    o_true = w_true @ vf[h].float()
                    w_sp = torch.softmax(s_true.masked_fill(~ms, -1e9), -1)
                    err = ((w_sp @ vf[h].float() - o_true).norm() / o_true.norm()).item()
                    res['route'].append(dict(t=t, g=g, li=li, h=h, fr=fr,
                                             cov_s=cov_s, cov_o=cov_o, iou=iou, err=err))
                del s_true, w_true, s_stale
            del ko
        # k-only reuse err (gap=1): attn(q,k_prev,v) vs attn(q,k,v)
        prev = torch.load(f'{OUT}/step_{t-1:03d}.pt', weights_only=True, map_location='cpu')
        kp = rh(prev[f'{li:02d}_k'], 0).to(DEV, DT)
        del prev
        for h in range(q.shape[0]):
            s1 = head_scores(q[h], kf[h]); s0 = head_scores(q[h], kp[h])
            o1 = torch.softmax(s1, -1) @ vf[h].float()
            o0 = torch.softmax(s0, -1) @ vf[h].float()
            res['kreuse'].append(dict(t=t, li=li, h=h,
                err=((o0 - o1).norm() / o1.norm()).item()))
        del q, kf, vf, kp
        torch.cuda.empty_cache()
    del cur
    print('anchor', t, flush=True)

json.dump(res, open(f'{OUT}/analysis.json', 'w'))
dk = np.mean([r['k'] for r in res['drift']])
dv = np.mean([r['v'] for r in res['drift'] if r['v']])
print(f'K drift={dk:.4f} V drift={dv:.4f} ratio={dk/dv:.3f}')
for fr in FRACS:
    rs = [r for r in res['route'] if r['fr'] == fr]
    print(f'frac={fr}: cov_s={np.mean([r["cov_s"] for r in rs]):.3f} '
          f'cov_o={np.mean([r["cov_o"] for r in rs]):.3f} '
          f'iou={np.mean([r["iou"] for r in rs]):.3f} err={np.mean([r["err"] for r in rs]):.3f}')
kr = np.mean([r['err'] for r in res['kreuse']])
print(f'k-reuse err={kr:.4f}')

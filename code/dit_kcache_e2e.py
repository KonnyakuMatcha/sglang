"""End-to-end K-cache validation on DiT-XL/2-256.
Modes:
  base    : vanilla
  k<N>    : reuse stale K (fp32), fresh V, refresh every N steps
  kq4<N>  : reuse stale K (4-bit per-channel), fresh V
  kv<N>   : reuse stale K+V
  feat<N> : cache whole block output (TeaCache-style)
Metrics: PSNR/SSIM vs base per seed; wall time per denoise step.
"""
import os, sys, json, time, math, argparse
import torch
import numpy as np
from PIL import Image

def quant_dequant_k(k, nbits=4):
    # k: (B,H,S,D) per-channel (per row d) symmetric
    qmax = 2 ** (nbits - 1) - 1
    scale = k.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    return (k / scale).round().clamp(-qmax, qmax) * scale

class CacheManager:
    def __init__(self, transformer, mode, N, qbits=None, use_cached_fn=None):
        self.mode = mode
        self.N = N
        self.qbits = qbits
        self.use_cached = use_cached_fn or (lambda step, layer: True)
        self.step = 0
        self.cur_step = -1
        self.k_cache = {}
        self.v_cache = {}
        self.feat_cache = {}
        self.attns = []
        self.blocks = []
        self.orig_procs = []
        self.orig_fwd = {}
        mgr = self

        for li, blk in enumerate(transformer.transformer_blocks):
            attn = blk.attn1
            self.attns.append(attn)
            self.orig_procs.append(attn.processor)
            self.blocks.append(blk)

            def make_proc(layer_idx, a):
                def proc(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
                    return mgr.attn_forward(attn, layer_idx, hidden_states, encoder_hidden_states, attention_mask)
                return proc
            attn.processor = make_proc(li, attn)

            if 'feat' in mode:
                of = blk.forward
                self.orig_fwd[li] = of
                def make_fwd(layer_idx, b):
                    def fwd(hidden_states, *args, **kwargs):
                        return mgr.block_forward(b, layer_idx, hidden_states, *args, **kwargs)
                    return fwd
                blk.forward = make_fwd(li, blk)

    def attn_forward(self, attn, layer, hs, enc_hs, attn_mask):
        residual = hs
        input_ndim = hs.ndim
        if input_ndim == 4:
            b, c, h, w = hs.shape
            hs = hs.view(b, c, h * w).transpose(1, 2)
        q = attn.to_q(hs)
        kv_src = enc_hs if enc_hs is not None else hs
        hit_k = self.use_cached(self.step, layer) and layer in self.k_cache and 'feat' not in self.mode
        hit_v = hit_k and 'kv' in self.mode and layer in self.v_cache
        fresh_k = None if hit_k else attn.to_k(kv_src)
        fresh_v = None if hit_v else attn.to_v(kv_src)
        heads, dh = attn.heads, attn.inner_dim // attn.heads
        b = q.shape[0]
        q = q.view(b, -1, heads, dh).transpose(1, 2)
        if fresh_k is not None:
            fresh_k = fresh_k.view(b, -1, heads, dh).transpose(1, 2)
        if fresh_v is not None:
            fresh_v = fresh_v.view(b, -1, heads, dh).transpose(1, 2)

        use_cache = hit_k
        use_kv = hit_v

        if use_cache:
            k = self.k_cache[layer]
        else:
            k = fresh_k
            if 'k' in self.mode or 'kv' in self.mode:
                store = quant_dequant_k(k, self.qbits) if (self.qbits and 'k' in self.mode) else k
                # store quantized K if qbits set for k-only mode; kv mode stores fp
                if 'kq' in self.mode:
                    store = quant_dequant_k(k, 4)
                self.k_cache[layer] = store.detach()
        if use_kv:
            v = self.v_cache[layer]
        else:
            v = fresh_v
            if 'kv' in self.mode:
                self.v_cache[layer] = v.detach()

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, attn.inner_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        if input_ndim == 4:
            out = out.transpose(1, 2).reshape(b, c, h, w)
        return out

    def block_forward(self, blk, layer, hs, *args, **kw):
        if self.use_cached(self.step, layer) and layer in self.feat_cache:
            return self.feat_cache[layer]
        out = self.orig_fwd[layer](hs, *args, **kw)
        self.feat_cache[layer] = out.detach() if torch.is_tensor(out) else out
        return out

    def detach(self):
        for a, p in zip(self.attns, self.orig_procs):
            a.processor = p
        for li, f in self.orig_fwd.items():
            self.blocks[li].forward = f

def build_step_counter(transformer, mgr):
    def pre_hook(module, args, kwargs):
        mgr.cur_step += 1
        mgr.step = mgr.cur_step
        return args, kwargs
    return transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)

def psnr_ssim(a, b):
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mse = ((a - b) ** 2).mean()
    psnr = 99 if mse == 0 else 10 * math.log10(1.0 / mse)
    # global SSIM (grayscale) approx
    ga, gb = a.mean(-1), b.mean(-1)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * ga.mean() * gb.mean() + C1) * (2 * ((ga * gb).mean() - ga.mean() * gb.mean()) + C2)) / \
           ((ga.mean() ** 2 + gb.mean() ** 2 + C1) * (ga.var() + gb.var() + C2))
    return psnr, float(ssim)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_images', type=int, default=64)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--out', default='/root/kv_e2e')
    ap.add_argument('--configs', default='base,k2,k4,k8,kq4,kv4,feat4')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from diffusers import DiTPipeline, DDIMScheduler
    pipe = DiTPipeline.from_pretrained('facebook/DiT-XL-2-256', torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to('cuda')
    tr = pipe.transformer
    tr.eval()

    n_batches = args.n_images // args.batch
    rng = np.random.RandomState(1234)
    class_pool = rng.choice(1000, size=args.batch, replace=False)
    seeds = list(range(n_batches))

    results = {}
    for cfg in args.configs.split(','):
        cfg = cfg.strip()
        mode, N, qb = 'base', 0, None
        if cfg != 'base':
            m = ''.join(c for c in cfg if c.isalpha())
            num = int(''.join(c for c in cfg if c.isdigit()))
            if m.startswith('kq'):
                mode, N = 'kq', num
            elif m == 'k':
                mode, N = 'k', num
            elif m == 'kv':
                mode, N = 'kv', num
            elif m == 'feat':
                mode, N = 'feat', num
        mgr, hook = None, None
        if mode != 'base':
            use_cached = lambda s, l: (s % N) != 0
            mgr = CacheManager(tr, mode, N, use_cached_fn=use_cached)
            hook = build_step_counter(tr, mgr)
        t0 = time.time()
        all_imgs = []
        for bi in range(n_batches):
            g = torch.Generator('cuda').manual_seed(seeds[bi])
            labels = class_pool.tolist()
            t_s = time.time()
            with torch.no_grad():
                out = pipe(class_labels=labels, generator=g, num_inference_steps=args.steps,
                           output_type='np')
            if mgr:
                mgr.cur_step = -1
            all_imgs.append((out.images * 255).round().astype(np.uint8))
            print(f'{cfg} batch{bi} {time.time()-t_s:.1f}s', flush=True)
        imgs = np.concatenate(all_imgs)
        np.save(f'{args.out}/imgs_{cfg}.npy', imgs)
        dt = time.time() - t0
        if cfg != 'base':
            base = np.load(f'{args.out}/imgs_base.npy')
            ps, ss = [], []
            for i in range(len(imgs)):
                p, s = psnr_ssim(imgs[i], base[i])
                ps.append(p); ss.append(s)
            results[cfg] = {'psnr': float(np.mean(ps)), 'ssim': float(np.mean(ss)), 'time': dt}
            print(cfg, results[cfg], flush=True)
        if mgr:
            mgr.detach()
        if hook:
            hook.remove()
    json.dump(results, open(f'{args.out}/e2e_metrics.json', 'w'), indent=2)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()

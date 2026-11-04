"""CogVideoX-2b e2e v3: all-heads-batched chunked sparse attention.

For each q-chunk (QC=1024), routing + attention done for all 30 heads in
one batched pass: (b,ih,QC,S) tensors, ~15GB transient.
"""
import os, sys, time, math, json
import torch, numpy as np
import torch.nn.functional as F
from diffusers import CogVideoXPipeline
from diffusers.models.embeddings import apply_rotary_emb

OUT = '/root/cogvid_e2e'
os.makedirs(OUT, exist_ok=True)
NUM_STEPS = 50
PROMPT = ("A golden retriever running on a beach at sunset, waves crashing, "
          "cinematic lighting, slow motion")
DEV = 'cuda'
QC = 1024

mode = sys.argv[1]
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
tag  = sys.argv[3] if len(sys.argv) > 3 else (mode if mode == 'base' else f'{mode}{int(frac*100)}')

def quant4(k):
    qmax = 7
    sc = k.abs().amax(-1, keepdim=True).clamp_min(1e-6).float() / qmax
    return ((k.float() / sc).round().clamp(-qmax, qmax) * sc)

pipe = CogVideoXPipeline.from_pretrained('THUDM/CogVideoX-2b',
                                         torch_dtype=torch.bfloat16)
pipe.to(DEV)
pipe.vae.enable_tiling()
tr = pipe.transformer
NL = len(tr.transformer_blocks)
kcache = {}

def make_proc(li):
    def proc(attn, hidden_states, encoder_hidden_states=None,
             attention_mask=None, image_rotary_emb=None):
        text_len = encoder_hidden_states.size(1)
        hs = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        b, s, _ = hs.shape
        ih = attn.heads
        dh = attn.to_q.out_features // ih
        q = attn.to_q(hs).view(b, s, ih, dh).transpose(1, 2)
        k = attn.to_k(hs).view(b, s, ih, dh).transpose(1, 2)
        v = attn.to_v(hs).view(b, s, ih, dh).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        if image_rotary_emb is not None:
            q[:, :, text_len:] = apply_rotary_emb(q[:, :, text_len:], image_rotary_emb)
            k[:, :, text_len:] = apply_rotary_emb(k[:, :, text_len:], image_rotary_emb)

        if mode == 'base':
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            kk = max(1, int(s * frac))
            kr = k if (mode == 'oracle' or li not in kcache) else kcache[li]
            qf, kf, krf, vf = q.float(), k.float(), kr.float(), v.float()
            out = torch.empty(b, ih, s, dh, device=q.device, dtype=torch.float32)
            scale = 1.0 / math.sqrt(dh)
            for i in range(0, s, QC):
                qc = qf[:, :, i:i+QC]                            # (b,ih,c,d)
                c = qc.shape[2]
                rs = torch.matmul(qc, krf.transpose(-1, -2)) * scale   # (b,ih,c,s)
                topi = rs.topk(kk, -1).indices
                del rs
                ss = torch.matmul(qc, kf.transpose(-1, -2)) * scale
                drop = torch.ones(b, ih, c, s, dtype=torch.bool, device=q.device)
                drop.scatter_(-1, topi, False)
                drop[:, :, :, :text_len] = False
                del topi
                w = torch.softmax(ss.masked_fill_(drop, float('-inf')), -1)
                del ss, drop
                out[:, :, i:i+QC] = torch.matmul(w, vf)
                del qc, w
            del qf, kf, krf, vf
            out = out.to(q.dtype)
            if mode == 'stale':
                kcache[li] = quant4(k.detach()).to(torch.bfloat16)
        out = out.transpose(1, 2).reshape(b, s, ih * dh)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        enc, hid = out.split([text_len, s - text_len], dim=1)
        return hid, enc
    return proc

for li in range(NL):
    tr.transformer_blocks[li].attn1.processor = make_proc(li)

g = torch.Generator(DEV).manual_seed(0)
t0 = time.time()
with torch.no_grad():
    frames = pipe(prompt=PROMPT, num_inference_steps=NUM_STEPS,
                  guidance_scale=6.0, generator=g, num_frames=49,
                  output_type='np').frames[0]
dt = time.time() - t0
np.save(f'{OUT}/imgs_{tag}.npy', frames)
print('SAVED', tag, frames.shape, f'{dt:.1f}s', flush=True)

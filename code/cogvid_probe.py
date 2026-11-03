"""CogVideoX-2b probe v2: capture POST-norm POST-RoPE Q/K (the tensors that
actually enter the score matmul), V, per selected layers.
K saved every step (bf16); Q,K,V saved at anchor steps only.
"""
import os, torch, numpy as np, json
import torch.nn.functional as F
from diffusers import CogVideoXPipeline
from diffusers.models.embeddings import apply_rotary_emb

OUT = '/root/cogvid_probe'
os.makedirs(OUT, exist_ok=True)
LAYERS = [8, 22]
ANCHORS = {5, 10, 15, 20, 25, 30, 35, 40, 45}
NUM_STEPS = 50

pipe = CogVideoXPipeline.from_pretrained('THUDM/CogVideoX-2b', torch_dtype=torch.bfloat16)
pipe.to('cuda')
pipe.vae.enable_tiling()
tr = pipe.transformer

store = {}
step_idx = [-1]
cur_step = [0]
print('heads', tr.config.num_attention_heads, 'dh', tr.config.attention_head_dim)

def prehook(mod, args, kwargs):
    step_idx[0] += 1
    cur_step[0] = step_idx[0]
    global store
    if store:
        torch.save(store, f'{OUT}/step_{step_idx[0]-1:03d}.pt')
    store = {}
    return args, kwargs

hooks = [tr.register_forward_pre_hook(prehook, with_kwargs=True)]

def make_proc(li):
    def proc(attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
             image_rotary_emb=None):
        text_len = encoder_hidden_states.size(1)
        hs = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        b, s, _ = hs.shape
        q = attn.to_q(hs); k = attn.to_k(hs); v = attn.to_v(hs)
        ih = attn.heads
        dh = k.shape[-1] // ih
        q = q.view(b, -1, ih, dh).transpose(1, 2)
        k = k.view(b, -1, ih, dh).transpose(1, 2)
        v = v.view(b, -1, ih, dh).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        if image_rotary_emb is not None:
            q[:, :, text_len:] = apply_rotary_emb(q[:, :, text_len:], image_rotary_emb)
            k[:, :, text_len:] = apply_rotary_emb(k[:, :, text_len:], image_rotary_emb)
        if cur_step[0] in ANCHORS:
            store[f'{li:02d}_q'] = q[:1].to(torch.bfloat16).cpu()
            store[f'{li:02d}_k'] = k[:1].to(torch.bfloat16).cpu()
            store[f'{li:02d}_v'] = v[:1].to(torch.bfloat16).cpu()
        else:
            store[f'{li:02d}_k'] = k[:1].to(torch.bfloat16).cpu()
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, ih * dh)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        enc, hid = out.split([text_len, out.size(1) - text_len], dim=1)
        return hid, enc
    return proc

for li in LAYERS:
    a = tr.transformer_blocks[li].attn1
    a.processor = make_proc(li)

g = torch.Generator('cuda').manual_seed(0)
prompt = ("A golden retriever running on a beach at sunset, waves crashing, "
          "cinematic lighting, slow motion")
with torch.no_grad():
    frames = pipe(prompt=prompt, num_inference_steps=NUM_STEPS,
                  guidance_scale=6.0, generator=g, num_frames=49,
                  output_type='np').frames[0]
torch.save(store, f'{OUT}/step_{step_idx[0]:03d}.pt')
np.save(f'{OUT}/frames.npy', frames)
print('captured', step_idx[0] + 1, 'steps; frames', frames.shape)

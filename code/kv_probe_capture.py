"""Capture per-step Q/K/V and block outputs from DiT-XL/2-256 for KV-reuse probing."""
import os, sys, json, time
import torch

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/root/kv_probe/captures"
MODEL = "facebook/DiT-XL-2-256"
NUM_STEPS = 50
CLASS_LABELS = [207, 360]
GUIDANCE = 4.0
SEED = 0
DTYPE = torch.float32

os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda"

from diffusers import DiTPipeline, DDIMScheduler

pipe = DiTPipeline.from_pretrained(MODEL, torch_dtype=DTYPE)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

tf = pipe.transformer
blocks = list(tf.transformer_blocks)
print(f"num blocks: {len(blocks)}", flush=True)

store = {}
_handles = []
step_counter = {"i": -1}
timetracker = {"ts": []}

def make_hook(li, name):
    def hook(module, args, output):
        store.setdefault(li, {})[name] = output.detach()
    return hook

for i, blk in enumerate(blocks):
    attn = blk.attn1
    _handles.append(attn.to_q.register_forward_hook(make_hook(i, "q")))
    _handles.append(attn.to_k.register_forward_hook(make_hook(i, "k")))
    _handles.append(attn.to_v.register_forward_hook(make_hook(i, "v")))
    _handles.append(attn.to_out[0].register_forward_hook(make_hook(i, "attn_out")))
    _handles.append(blk.register_forward_hook(make_hook(i, "block_out")))

def flush_store():
    i = step_counter["i"]
    step_dir = os.path.join(OUT_DIR, f"step_{i:03d}")
    os.makedirs(step_dir, exist_ok=True)
    pack = {}
    for li, d in store.items():
        for k, v in d.items():
            pack[f"{li:02d}_{k}"] = v.to(torch.bfloat16).cpu()
    torch.save(pack, os.path.join(step_dir, "tensors.pt"))
    store.clear()

def tf_prehook(module, args, kwargs=None):
    # new denoising step: flush previous step's store first
    if store:
        flush_store()
    step_counter["i"] += 1

def tf_posthook(module, args, kwargs, output):
    pass

tf.register_forward_pre_hook(tf_prehook, with_kwargs=True)

gen = torch.Generator(device=device).manual_seed(SEED)
t0 = time.time()
out = pipe(
    class_labels=CLASS_LABELS,
    guidance_scale=GUIDANCE,
    num_inference_steps=NUM_STEPS,
    generator=gen,
    output_type="np",
)
if store:
    flush_store()
print(f"inference done in {time.time()-t0:.1f}s", flush=True)

meta = dict(model=MODEL, steps=NUM_STEPS, class_labels=CLASS_LABELS,
            guidance=GUIDANCE, seed=SEED,
            timesteps=pipe.scheduler.timesteps.cpu().tolist())
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
from PIL import Image
import numpy as np
img = (out.images[0] * 255).astype(np.uint8) if out.images[0].max() <= 1 else out.images[0].astype(np.uint8)
Image.fromarray(img).save(os.path.join(OUT_DIR, "sample.png"))
print("saved to", OUT_DIR)

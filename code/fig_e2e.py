import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

m = json.load(open('/root/kv_e2e/e2e_metrics.json'))
rows = [
    ('K-only, N=2', m['k2']['psnr'], m['k2']['ssim']),
    ('K-only, N=4', m['k4']['psnr'], m['k4']['ssim']),
    ('K-only, N=8', m['k8']['psnr'], m['k8']['ssim']),
    ('K-4bit, N=4', m['kq4']['psnr'], m['kq4']['ssim']),
    ('KV, N=2', m['kv2']['psnr'], m['kv2']['ssim']),
    ('Feature, N=2', m['feat2']['psnr'], m['feat2']['ssim']),
    ('Feature, N=4', m['feat4']['psnr'], m['feat4']['ssim']),
]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
names = [r[0] for r in rows]
colors = ['#2a7f62']*3 + ['#3aa08a'] + ['#c06a4d'] + ['#5b6abf']*2
ax[0].barh(names[::-1], [r[1] for r in rows][::-1], color=colors[::-1])
ax[0].set_xlabel('PSNR vs. exact (dB)')
ax[0].set_title('DiT-XL/2 256x256, 50 DDIM steps')
ax[1].barh(names[::-1], [r[2] for r in rows][::-1], color=colors[::-1])
ax[1].set_xlabel('SSIM vs. exact')
ax[1].set_xlim(0.8, 1.0)
for a in ax:
    a.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/root/kv_e2e/fig_e2e.png', dpi=140)
print('saved')

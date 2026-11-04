import numpy as np, os, json, torch
import torch.nn.functional as F

OUT = '/root/cogvid_e2e'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
base = torch.from_numpy(np.load(f'{OUT}/imgs_base.npy')).to(DEV, torch.float64)
# (F,H,W,3) -> (F,3,H,W)
base = base.permute(0, 3, 1, 2)

def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return -10 * np.log10(mse) if mse > 0 else 99.0

def ssim_frame(x, y, C1=0.01**2, C2=0.03**2, win=7):
    """x,y: (3,H,W). uniform-window SSIM."""
    x = x.unsqueeze(0); y = y.unsqueeze(0)
    mx = F.avg_pool2d(x, win, 1, win//2)
    my = F.avg_pool2d(y, win, 1, win//2)
    sxx = F.avg_pool2d(x*x, win, 1, win//2) - mx*mx
    syy = F.avg_pool2d(y*y, win, 1, win//2) - my*my
    sxy = F.avg_pool2d(x*y, win, 1, win//2) - mx*my
    m = ((2*mx*my + C1)*(2*sxy + C2)) / ((mx*mx + my*my + C1)*(sxx + syy + C2))
    return m.mean().item()

res = {}
mj = f'{OUT}/e2e_metrics.json'
if os.path.exists(mj):
    res = json.load(open(mj))

for f in sorted(os.listdir(OUT)):
    if not (f.startswith('imgs_') and f.endswith('.npy')) or f == 'imgs_base.npy':
        continue
    tag = f[5:-4]
    vid = torch.from_numpy(np.load(f'{OUT}/{f}')).to(DEV, torch.float64).permute(0, 3, 1, 2)
    ps = [psnr(base[i], vid[i]) for i in range(len(vid))]
    ss = [ssim_frame(base[i], vid[i]) for i in range(len(vid))]
    res[tag] = {'psnr': float(np.mean(ps)), 'ssim': float(np.mean(ss)),
                'psnr_min': float(np.min(ps)), 'n_frames': len(vid)}
    print(f'{tag}: PSNR={np.mean(ps):.2f} (min {np.min(ps):.2f}) SSIM={np.mean(ss):.4f}')

json.dump(res, open(mj, 'w'), indent=2)

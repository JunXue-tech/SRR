"""Estimate per-frequency response normalization from training pairs only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
from teacher import SCALES, ResponseSpectra


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    frontend = ResponseSpectra().to(a.device).eval()
    sums = [torch.zeros(2, n//2+1, dtype=torch.float64, device=a.device) for n, _ in SCALES]
    squares = [torch.zeros_like(x) for x in sums]; counts = [0]*len(SCALES)
    with torch.no_grad():
        for line in Path(a.manifest).open():
            if not line.strip(): continue
            row = json.loads(line)
            x, rate = torchaudio.load(row['original_path'], backend='ffmpeg')
            sr, high_rate = torchaudio.load(row['superres_path'], backend='ffmpeg')
            n = row['valid_original_samples']
            assert rate == 16000 and high_rate == 48000 and row['valid_superres_samples'] == 3*n
            x = x[:,:n].to(a.device); sr = sr[:,:3*n].to(a.device)
            for i, (feature, valid) in enumerate(frontend(x, sr, torch.tensor([3*n], device=a.device))):
                values = feature[0,:2,:,valid[0]].double()
                sums[i] += values.sum(-1); squares[i] += values.square().sum(-1); counts[i] += values.shape[-1]
    result = {}
    for i in range(len(SCALES)):
        if counts[i] == 0: raise ValueError('Empty training manifest')
        mean = sums[i]/counts[i]
        std = (squares[i]/counts[i]-mean.square()).clamp_min(1e-8).sqrt()
        result[f'mean_{i}'] = mean.float().cpu().numpy()
        result[f'std_{i}'] = std.float().cpu().numpy()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, **result)

if __name__ == '__main__':
    main()

"""Prepare aligned 16/48-kHz pairs using the official FlashSR implementation."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import torch
import torchaudio
import soundfile as sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--flashsr-root', required=True, help='Official FlashSR_Inference checkout')
    p.add_argument('--weights-dir', required=True, help='student_ldm.pth, sr_vocoder.pth, vae.pth')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--split', default='train')
    p.add_argument('--first-only', action='store_true', help='Use for development/evaluation')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    root = Path(a.flashsr_root).resolve()
    weights = Path(a.weights_dir).resolve()
    rows = [json.loads(x) for x in Path(a.manifest).open() if x.strip()]
    out = Path(a.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = out / (a.split + '.jsonl')
    if manifest.exists():
        raise FileExistsError('Choose a fresh output split: ' + str(manifest))
    if len({r['utterance_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate utterance IDs')
    # Resolve input paths before changing to the upstream repository's working directory.
    for row in rows:
        row['audio_path'] = str(Path(row['audio_path']).resolve())
    sys.path.insert(0, str(root)); os.chdir(root)
    from FlashSR.FlashSR import FlashSR
    device = torch.device('cuda')
    model = FlashSR(str(weights/'student_ldm.pth'), str(weights/'sr_vocoder.pth'), str(weights/'vae.pth'))
    model = model.to(device).eval()
    model.vae.to(device); model.vae.autoencoder.eval()
    resample = torchaudio.transforms.Resample(16000, 48000).to(device)
    with manifest.with_suffix('.jsonl.tmp').open('w') as f, torch.inference_mode():
        for row in rows:
            wave, rate = torchaudio.load(row['audio_path'], backend='ffmpeg'); wave = wave.mean(0)
            if rate != 16000:
                wave = torchaudio.functional.resample(wave, rate, 16000)
            if not len(wave) or not torch.isfinite(wave).all() or row['label'] not in (0, 1):
                raise ValueError('Invalid input: ' + row['utterance_id'])
            n = len(wave)
            starts = [0] if a.first_only or n <= 81920 else list(range(0, n-81920+1, 40960))
            if not a.first_only and n > 81920 and starts[-1] != n-81920:
                starts.append(n-81920)
            folder = out/a.split/hashlib.sha256(row['utterance_id'].encode()).hexdigest()
            folder.mkdir(parents=True, exist_ok=True)
            for i, start in enumerate(starts):
                piece = wave[start:start+81920]; valid = len(piece)
                piece = torch.nn.functional.pad(piece, (0, 81920-valid))
                seed = int(hashlib.sha256(f'{a.seed}|{row["utterance_id"]}|{i}'.encode()).hexdigest()[:8], 16)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                result = model(resample(piece[None].to(device)), num_steps=1, lowpass_input=False)
                if result.shape != (1, 245760) or not torch.isfinite(result).all():
                    raise ValueError('Invalid FlashSR output')
                original = folder/f'{i:04d}.original.wav'; superres = folder/f'{i:04d}.superres.wav'
                sf.write(original, piece.numpy(), 16000, subtype='FLOAT')
                sf.write(superres, result[0].float().cpu().numpy(), 48000, subtype='PCM_16')
                record = dict(utterance_id=row['utterance_id'], label=row['label'], window_index=i,
                              original_path=str(original), superres_path=str(superres),
                              valid_original_samples=valid, valid_superres_samples=3*valid)
                f.write(json.dumps(record)+'\n'); f.flush()
            print(row['utterance_id'], flush=True)
    manifest.with_suffix('.jsonl.tmp').replace(manifest)

if __name__ == '__main__':
    main()

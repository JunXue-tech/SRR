"""Aligned random training crops and first-crop evaluation (real=0, fake=1)."""
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, Sampler
from rawboost import rawboost
SIZE = 64600

class MatchedDataset(Dataset):
    def __init__(self, manifest, training=True, needs_sr=True):
        self.needs_sr = needs_sr
        if not training:
            raise ValueError("Use RawEvalDataset for evaluation")
        self.manifest = Path(manifest)
        self.rows = [json.loads(l) for l in self.manifest.open() if l.strip()]
        groups = {}
        for i, r in enumerate(self.rows):
            groups.setdefault(r["utterance_id"], []).append(i)
        self.groups = [groups[k] for k in sorted(groups)]
        self.labels = [self.rows[g[0]]["label"] for g in self.groups]
        self.algorithm = int(os.environ["RAWBOOST_ALGO"])
        self.augmentation_probability = float(os.environ["RAWBOOST_PROBABILITY"])
        assert self.algorithm in (4, 5) and 0 <= self.augmentation_probability <= 1

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        weight = 1.0
        if isinstance(index, tuple):
            index, weight = index
        i = random.choice(self.groups[index])
        r = self.rows[i]
        n = int(r["valid_original_samples"])
        m = int(r["valid_superres_samples"])
        assert 0 < n <= 81920 and m == 3 * n
        start = random.randint(0, max(0, n - SIZE))
        n = min(SIZE, n - start)
        m = 3 * n
        # Same physical crop; augmentation draws and normalization are independent.
        raw = self.load_audio(i, "original", 16000)[start : start + n].copy()
        sr = (
            self.load_audio(i, "superres", 48000)[3 * start : 3 * start + m].copy()
            if self.needs_sr
            else np.zeros(1, dtype=np.float32)
        )
        # One clean/augmented decision per example; independent parameters per view.
        # A clean example preserves both original and pre-generated SR waveforms.
        if random.random() < self.augmentation_probability:
            raw = rawboost(raw, 16000, self.algorithm)
            if self.needs_sr:
                sr = rawboost(sr, 48000, self.algorithm)
        raw = torch.nn.functional.pad(torch.from_numpy(raw), (0, SIZE - n))
        sr = (
            torch.nn.functional.pad(torch.from_numpy(sr), (0, 3 * SIZE - m))
            if self.needs_sr
            else torch.zeros(1)
        )
        return raw, n, sr, m, int(r["label"]), float(weight)

    def load_audio(self, index, key, rate):
        value, sr = torchaudio.load(self.rows[index][key + "_path"], backend="ffmpeg")
        if sr != rate or value.shape[0] != 1:
            raise ValueError("Cache must be native-rate mono")
        if value.shape[1] < self.rows[index]["valid_" + key + "_samples"]:
            raise ValueError("Truncated cache")
        return value[0].numpy()


class RawEvalDataset(Dataset):
    """Exactly one leading 64,600-sample crop per utterance; pad short inputs."""
    def __init__(self, manifest):
        self.rows = [json.loads(line) for line in Path(manifest).open() if line.strip()]
        if not self.rows or len({r["utterance_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Manifest must contain unique, nonempty utterances")
        self.labels = [r["label"] for r in self.rows]
        if any(y not in (0, 1) for y in self.labels):
            raise ValueError("Labels must be 0 (real) or 1 (fake)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        wave, rate = torchaudio.load(row["audio_path"], backend="ffmpeg")
        wave = wave.mean(0)
        if rate != 16000:
            wave = torchaudio.functional.resample(wave, rate, 16000)
        piece = wave[:SIZE]
        n = len(piece)
        if n == 0 or not torch.isfinite(piece).all():
            raise ValueError("Empty or nonfinite audio: " + row["utterance_id"])
        return F.pad(piece, (0, SIZE-n)), n, row["label"], row["utterance_id"], 0


class PairedEval(RawEvalDataset):
    """Match the first original crop with the first FlashSR tile, without blending."""
    def __init__(self, manifest, paired_manifest):
        super().__init__(manifest)
        if not paired_manifest:
            raise ValueError("Teacher evaluation requires --sr-manifest")
        self.pairs = {}
        for line in Path(paired_manifest).open():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("window_index", 0) == 0:
                uid = row["utterance_id"]
                if uid in self.pairs:
                    raise ValueError("Duplicate first SR tile: " + uid)
                self.pairs[uid] = row
        if set(self.pairs) != {r["utterance_id"] for r in self.rows}:
            raise ValueError("Original and super-resolution manifests do not match")

    def __getitem__(self, index):
        original, n, label, uid, _ = super().__getitem__(index)
        pair = self.pairs[uid]
        raw, rate = torchaudio.load(pair["original_path"], backend="ffmpeg")
        sr, high_rate = torchaudio.load(pair["superres_path"], backend="ffmpeg")
        if rate != 16000 or high_rate != 48000 or raw.shape[0] != 1 or sr.shape[0] != 1:
            raise ValueError("Paired cache must be mono at 16/48 kHz")
        if pair["label"] != label or pair["valid_original_samples"] < n:
            raise ValueError("Misaligned cache labels or lengths")
        if not torch.allclose(raw[0, :n], original[:n], atol=4e-5, rtol=1e-4):
            raise ValueError("Original crop and FlashSR conditioning do not match: " + uid)
        piece = sr[0, :3*n]
        if len(piece) != 3*n or not torch.isfinite(piece).all():
            raise ValueError("Invalid super-resolution crop")
        return original, n, F.pad(piece, (0, 3*SIZE-3*n)), 3*n, label, uid, 0, 0


class GlobalBatchSampler(Sampler):
    """Exactly ``batch`` nonzero-weight examples per optimizer update.

    Padded slots synchronize ranks but contribute zero gradient. Global draw is
    balanced over the epoch, matching the previous replacement-sampling policy.
    """

    def __init__(self, labels, rank, world, micro, seed, batch=32):
        self.rank, self.world, self.micro, self.seed, self.batch = rank, world, micro, seed, batch
        self.accumulation = math.ceil(batch / (world * micro))
        self.slots = self.accumulation * world * micro
        self.updates = math.ceil(len(labels) / batch)
        self.sources = [torch.tensor([i for i, v in enumerate(labels) if v == k]) for k in (0, 1)]
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.updates * self.accumulation

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        count = self.updates * self.batch
        selected = torch.cat(
            [s[torch.randint(len(s), (count // 2,), generator=g)] for s in self.sources]
        )
        selected = selected[torch.randperm(count, generator=g)]
        for update in range(self.updates):
            indices = selected[update * self.batch : (update + 1) * self.batch].tolist()
            # Rotate zero-weight padding across ranks without altering the batch.
            pairs = [(i, 1.0) for i in indices] + [(indices[0], 0.0)] * (self.slots - self.batch)
            offset = update % self.world
            for step in range(self.accumulation):
                start = (
                    step * self.world * self.micro
                    + ((self.rank + offset) % self.world) * self.micro
                )
                yield pairs[start : start + self.micro]


class DistributedEvalSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterable[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return max(0, (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size)



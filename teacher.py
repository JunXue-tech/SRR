"""Teacher: DGSE + MSRM + RCDR + concatenation classification.

MSRM: Multiscale Response Modeling. FlashSR is frozen and runs offline.
The response-only auxiliary head is trained under TeacherLoss.
When shared_encoder=True, joint training supplies detached DGSE streams.
Standalone teacher training owns and optimizes its own DGSE.
"""

import math
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from student import DGSE, RCDR, AttentivePool, LocalCrossAttention

SCALES = [(1024, 960), (2048, 1920), (4096, 3840)]
BANDS = [(0, 2000), (2000, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, 24000)]


class ResponseSpectra(nn.Module):
    def __init__(self, stats_path=None):
        super().__init__()
        self.upsample = torchaudio.transforms.Resample(16000, 48000)
        stats = np.load(stats_path) if stats_path else None
        for i, (fft, win) in enumerate(SCALES):
            self.register_buffer(f"window_{i}", torch.hann_window(win), persistent=False)
            mask = (torch.arange(fft // 2 + 1) * 48000 / fft <= 8000).float()
            self.register_buffer(f"band_mask_{i}", mask, persistent=False)
            mean = torch.from_numpy(stats[f"mean_{i}"]).float() if stats is not None else torch.zeros(2, fft // 2 + 1)
            std = torch.from_numpy(stats[f"std_{i}"]).float() if stats is not None else torch.ones(2, fft // 2 + 1)
            self.register_buffer(f"mean_{i}", mean)
            self.register_buffer(f"std_{i}", std.clamp_min(1e-4))
        self.has_stats = True

    def forward(self, original, superres, lengths):
        with torch.autocast(device_type=original.device.type, enabled=False):
            reference = self.upsample(original.float())
            if reference.shape != superres.shape:
                raise ValueError("16k/48k padded lengths must match exactly 1:3")
            results = []
            for i, (fft, win) in enumerate(SCALES):
                args = dict(
                    n_fft=fft,
                    hop_length=480,
                    win_length=win,
                    window=getattr(self, f"window_{i}"),
                    center=True,
                    pad_mode="constant",
                    return_complex=True,
                )
                high = torch.stft(superres.float(), **args).abs().add(1e-5).log()
                low = torch.stft(reference, **args).abs().add(1e-5).log()
                band = getattr(self, f"band_mask_{i}")[None, :, None]
                pair = torch.stack([high, (high - low) * band], dim=1)
                valid = (
                    torch.arange(pair.shape[-1], device=pair.device)[None, :] * 480
                    <= lengths[:, None]
                )
                if self.has_stats:
                    pair = (pair - getattr(self, f"mean_{i}")[None, :, :, None]) / getattr(
                        self, f"std_{i}"
                    )[None, :, :, None]
                feature = torch.cat(
                    [pair, band[:, None].expand(pair.shape[0], 1, -1, pair.shape[-1])], dim=1
                )
                results.append((feature.masked_fill(~valid[:, None, None, :], 0), valid))
            return results


class ResidualConv(nn.Module):
    def __init__(self, inc, outc, stride=2):
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(inc, outc, 3, stride=(stride, 1), padding=1, bias=False),
            nn.GroupNorm(8, outc),
            nn.GELU(),
            nn.Conv2d(outc, outc, 3, padding=1, bias=False),
            nn.GroupNorm(8, outc),
        )
        self.skip = nn.Conv2d(inc, outc, 1, stride=(stride, 1), bias=False)

    def forward(self, x):
        return F.gelu(self.path(x) + self.skip(x))


class AxialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_norm = nn.LayerNorm(64)
        self.band = nn.MultiheadAttention(64, 4, dropout=0.1, batch_first=True)
        self.time_norm = nn.LayerNorm(64)
        self.time = LocalCrossAttention(64, 64, 4, 4)
        self.ff_norm = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 64))

    def forward(self, x, valid):
        b, t, k, d = x.shape
        n = self.band_norm(x).reshape(b * t, k, d)
        x = x + self.band(n, n, n, need_weights=False)[0].reshape(b, t, k, d)
        n = self.time_norm(x).permute(0, 2, 1, 3).reshape(b * k, t, d)
        mask = valid[:, None, :].expand(-1, k, -1).reshape(b * k, t)
        update = self.time(n, n[:, :, None, :], mask).reshape(b, k, t, d).permute(0, 2, 1, 3)
        x = x + update
        x = x + self.ff(self.ff_norm(x))
        return x.masked_fill(~valid[:, :, None, None], 0)


class MSRM(nn.Module):
    """Multiscale Response Modeling: paired waveforms -> features and sequence."""

    def __init__(self, stats_path, checkpoint_encoder=True):
        super().__init__()
        self.frontend = ResponseSpectra(stats_path)
        self.checkpoint_encoder = checkpoint_encoder
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=(2, 1), padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResidualConv(32, 64),
            ResidualConv(64, 64),
        )
        self.band_projection = nn.Sequential(nn.Linear(4 * 64, 64), nn.LayerNorm(64), nn.GELU())
        self.scale_embedding = nn.Parameter(torch.randn(3, 64) * 0.02)
        self.band_embedding = nn.Parameter(torch.randn(6, 64) * 0.02)
        self.scale_score = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.axial = nn.ModuleList([AxialBlock(), AxialBlock()])
        self.sequence = nn.Sequential(nn.Linear(384, 256), nn.LayerNorm(256))

    def encode_scale(self, feature, scale):
        value = self.stem(feature)
        fft = SCALES[scale][0]
        tokens = []
        for low, high in BANDS:
            left = int(math.ceil(low * fft / (48000 * 8)))
            right = value.shape[2] if high == 24000 else int(math.ceil(high * fft / (48000 * 8)))
            block = F.adaptive_avg_pool2d(
                value[:, :, left : max(left + 1, right), :], (4, value.shape[-1])
            )
            block = block.permute(0, 3, 2, 1).flatten(2)
            tokens.append(self.band_projection(block))
        return torch.stack(tokens, 2) + self.scale_embedding[scale]

    def forward(self, original, superres, lengths, target_frames, valid):
        spectra = self.frontend(original, superres, lengths)
        features = []
        for scale, (feature, _) in enumerate(spectra):
            value = (
                checkpoint(self.encode_scale, feature, scale, use_reentrant=False)
                if self.training and self.checkpoint_encoder
                else self.encode_scale(feature, scale)
            )
            features.append(value)
        multiscale = torch.stack(features, dim=3)
        weight = self.scale_score(multiscale).float().softmax(dim=3).to(multiscale.dtype)
        tokens = (weight * multiscale).sum(3)
        # STFT centers are n*10ms; XLS-R RF=400 samples, stride=320 at 16 kHz.
        position = (199.5 + torch.arange(target_frames, device=tokens.device) * 320) * 3 / 480
        lower = position.floor().long().clamp(0, tokens.shape[1] - 1)
        upper = (lower + 1).clamp_max(tokens.shape[1] - 1)
        fraction = (position - position.floor())[None, :, None, None].to(tokens.dtype)
        tokens = tokens[:, lower] * (1 - fraction) + tokens[:, upper] * fraction
        tokens = (tokens + self.band_embedding[None, None]).masked_fill(~valid[:, :, None, None], 0)
        for block in self.axial:
            tokens = block(tokens, valid)
        return tokens, self.sequence(tokens.flatten(2)).masked_fill(~valid[:, :, None], 0)


class FusionClassifier(nn.Sequential):
    """Fusion & Classifier: speech, response, absolute difference, elementwise product."""

    def __init__(self):
        super().__init__(nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1))


class TeacherModel(nn.Module):
    def __init__(self, config, shared_encoder=False):
        super().__init__()
        self.dgse = None if shared_encoder else DGSE(config["paths"]["xlsr_checkpoint"], config["training"].get("xlsr_output_layer",23))
        self.checkpoint_xlsr = config["training"]["xlsr_gradient_checkpointing"]
        self.msrm = MSRM(
            config["paths"]["response_stats"], config["training"]["spectral_gradient_checkpointing"]
        )
        self.rcdr = RCDR()
        self.response_pool = AttentivePool()
        self.response_classifier = nn.Linear(256, 1)
        self.classifier = FusionClassifier()

    def forward(self, original, lengths, superres, superres_lengths, streams=None, valid=None):
        original = original.masked_fill(
            torch.arange(original.shape[1], device=original.device)[None] >= lengths[:, None], 0
        )
        superres = superres.masked_fill(
            torch.arange(superres.shape[1], device=superres.device)[None]
            >= superres_lengths[:, None],
            0,
        )
        if streams is None:
            if self.dgse is None:
                raise ValueError("Shared teacher requires DGSE streams")
            if self.training and self.checkpoint_xlsr:
                streams, valid = checkpoint(self.dgse, original, lengths, use_reentrant=False)
            else:
                streams, valid = self.dgse(original, lengths)
        elif valid is None:
            raise ValueError("Missing DGSE valid-frame mask")
        features, sequence = self.msrm(
            original, superres, superres_lengths, streams.shape[1], valid
        )
        speech, reads, attention = self.rcdr(streams, features, valid)
        response, _ = self.response_pool(sequence, valid)
        fused = torch.cat([speech, response, (speech-response).abs(), speech*response], dim=-1)
        return dict(
            teacher_logit=self.classifier(fused).squeeze(-1),
            superres_logit=self.response_classifier(response).squeeze(-1),
            teacher_tokens=features,
            teacher_reads=reads,
            teacher_attention=attention,
            valid=valid,
        )

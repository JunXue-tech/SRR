"""Student model and shared speech/routing primitives.

DGSE: Depth-Grouped Speech Encoder.
RPM: Response Prediction Module.
RCDR: Response-Conditioned Dynamic Routing.
Teacher imports shared primitives from here; neither branch duplicates DGSE
parameters during joint training. Student inference never imports the SR frontend.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _legacy_numpy_aliases():
    for name, value in {"float": float, "int": int}.items():
        if name not in np.__dict__:
            setattr(np, name, value)


class DGSE(nn.Module):
    """Depth-Grouped Speech Encoder: B,N -> B,T,4,256 with valid-frame mask."""

    def __init__(self, path, output_layer=None):
        super().__init__()
        _legacy_numpy_aliases()
        import fairseq

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
        self.ssl = models[0]
        self.ssl.encoder.layerdrop = 0.0
        self.ssl.feature_grad_mult = 1.0
        self.output_layer = output_layer
        unused = ["quantizer.", "project_q.", "final_proj."]
        # With pre-LN, final encoder norm affects returned x, not the collected layer states.
        if self.ssl.encoder.layer_norm_first:
            unused.append("encoder.layer_norm.")
        self.inactive_parameters = []
        for name, p in self.ssl.named_parameters():
            if name == "mask_emb" or name.startswith(tuple(unused)):
                p.requires_grad_(False)
                self.inactive_parameters.append(name)
        self.layer_logits = nn.Parameter(torch.zeros(4, 6))
        self.projections = nn.ModuleList(
            [nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.LayerNorm(256)) for _ in range(4)]
        )

    def forward(self, x, lengths):
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        x = x.float() * mask
        mean = x.sum(1, keepdim=True) / lengths[:, None]
        centered = (x - mean) * mask
        std = (centered.square().sum(1, keepdim=True) / lengths[:, None]).sqrt().clamp_min(1e-5)
        normalized = centered / std
        result = self.ssl(
            normalized, padding_mask=~mask, mask=False, features_only=True, layer=self.output_layer
        )
        layers = result["layer_results"]
        if len(layers) != 24:
            raise RuntimeError(f"Expected 24 XLS-R states; got {len(layers)}")
        weights = self.layer_logits.softmax(-1)
        streams = []
        for g in range(4):
            state = sum(
                weights[g, j].to(layers[g * 6 + j][0].dtype) * layers[g * 6 + j][0].transpose(0, 1)
                for j in range(6)
            )
            streams.append(self.projections[g](state))
        streams = torch.stack(streams, dim=2)
        frame_lengths = self.ssl._get_feat_extract_output_lengths(lengths).clamp(
            min=1, max=streams.shape[1]
        )
        valid = torch.arange(streams.shape[1], device=x.device)[None, :] < frame_lengths[:, None]
        return streams.masked_fill(~valid[:, :, None, None], 0), valid


class LocalCrossAttention(nn.Module):
    def __init__(self, dim, kv_dim, heads, radius, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.radius = radius
        self.dropout = dropout
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, query, tokens, valid):
        b, t, d = query.shape
        k = tokens.shape[2]
        h = self.heads
        dh = d // h
        offsets = torch.arange(-self.radius, self.radius + 1, device=query.device)
        indices = torch.arange(t, device=query.device)[:, None] + offsets[None, :]
        inside = (indices >= 0) & (indices < t)
        indices = indices.clamp(0, t - 1)
        q = self.q(query).reshape(b * t, h, 1, dh)

        def project(module):
            v = module(tokens).reshape(b, t, k, h, dh)[:, indices]
            return v.permute(0, 1, 4, 2, 3, 5).reshape(b * t, h, -1, dh)

        keys = project(self.k)
        values = project(self.v)
        allowed = valid[:, indices] & inside[None]
        # Invalid queries are zeroed afterwards; give them a defined softmax row.
        allowed = allowed | (~valid[:, :, None] & (offsets[None, None, :] == 0))
        allowed = allowed[:, :, :, None].expand(-1, -1, -1, k).reshape(b * t, 1, 1, -1)
        out = F.scaled_dot_product_attention(
            q, keys, values, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0
        )
        return self.output(out.reshape(b, t, d)).masked_fill(~valid[:, :, None], 0)


class GuidedResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.stream_norm = nn.LayerNorm(256)
        self.token_norm = nn.LayerNorm(64)
        self.controller = nn.Sequential(nn.Linear(256 + 384, 128), nn.GELU(), nn.Linear(128, 24))
        nn.init.normal_(self.controller[-1].weight, std=1e-3)
        nn.init.zeros_(self.controller[-1].bias)
        self.cross = LocalCrossAttention(256, 64, 4, 2)
        self.ff_norm = nn.LayerNorm(256)
        self.ff = nn.Sequential(
            nn.Linear(256, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 256)
        )

    @staticmethod
    def mixing(logits):
        logp = logits.float()
        for _ in range(10):
            logp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)
            logp = logp - torch.logsumexp(logp, dim=-2, keepdim=True)
        return 0.9 * torch.eye(4, device=logits.device) + 0.1 * logp.exp()


class RoutedBlock(GuidedResidualBlock):
    def forward(self, streams, tokens, valid):
        normalized = self.stream_norm(streams)
        q = self.token_norm(tokens)
        control = self.controller(torch.cat([normalized.mean(2), q.flatten(2)], -1))
        matrix = self.mixing(control[..., :16].reshape(*control.shape[:2], 4, 4))
        read = control[..., 16:20].float().softmax(-1)
        write = control[..., 20:24].sigmoid()
        query = (normalized * read.to(streams.dtype)[..., None]).sum(2)
        update = self.cross(query, q, valid)
        update = update + self.ff(self.ff_norm(query + update))
        residual = torch.einsum("btji,btid->btjd", matrix.to(streams.dtype), streams)
        result = residual + 0.1 * write[..., None] * update[:, :, None, :]
        return result.masked_fill(~valid[:, :, None, None], 0), read


class AttentivePool(nn.Module):
    """Temporal attention, weighted mean/std, and 512-to-256 projection."""
    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(256,128), nn.Tanh(), nn.Linear(128,1))
        self.projection = nn.Sequential(nn.Linear(512,256), nn.LayerNorm(256))
    def forward(self, x, valid):
        scores = self.attention(x).squeeze(-1).float().masked_fill(~valid, -1e9)
        weights = scores.softmax(1)
        mean = (weights[...,None] * x.float()).sum(1)
        std = (weights[...,None] * (x.float()-mean[:,None]).square()).sum(1).clamp_min(1e-5).sqrt()
        return self.projection(torch.cat([mean,std],-1)), weights


class RCDR(nn.Module):
    """Response-Conditioned Dynamic Routing: read, mix, cross-attention, write."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([RoutedBlock(), RoutedBlock()])
        self.readout = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1))
        self.pool = AttentivePool()

    def forward(self, streams, tokens, valid):
        reads = []
        for block in self.blocks:
            streams, read = block(streams, tokens, valid)
            reads.append(read)
        weight = self.readout(streams).float().softmax(2).to(streams.dtype)
        vector, attention = self.pool((weight * streams).sum(2), valid)
        return vector, torch.stack(reads, 1), attention


class RPM(nn.Module):
    """Response Prediction Module: speech streams -> B,T,6,64 SR-response features."""

    def __init__(self, dilation=1, residual=False):
        super().__init__()
        self.residual = residual
        self.input = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256))
        self.depthwise = nn.Conv1d(256, 256, 5, padding=2 * dilation, dilation=dilation, groups=256)
        self.pointwise = nn.Conv1d(256, 256, 1)
        self.output = nn.Linear(256, 384)

    def forward(self, streams, valid):
        x = self.input(streams.flatten(2)).masked_fill(~valid[..., None], 0)
        update = F.gelu(self.pointwise(self.depthwise(x.transpose(1, 2)))).transpose(1, 2)
        x = x + update if self.residual else update
        return self.output(x).reshape(*x.shape[:2], 6, 64).masked_fill(~valid[:, :, None, None], 0)


class StudentClassifier(nn.Sequential):
    """Student classifier; RCDR performs attentive temporal statistics pooling."""

    def __init__(self):
        super().__init__(nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1))


class StudentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dgse = DGSE(config["paths"]["xlsr_checkpoint"], config["training"].get("xlsr_output_layer",23))
        self.checkpoint_xlsr = config["training"]["xlsr_gradient_checkpointing"]
        self.rpm = RPM()
        self.rcdr = RCDR()
        self.classifier = StudentClassifier()

    def forward(self, original, lengths):
        original = original.masked_fill(
            torch.arange(original.shape[1], device=original.device)[None] >= lengths[:, None], 0
        )
        if self.training and self.checkpoint_xlsr:
            streams, valid = checkpoint(self.dgse, original, lengths, use_reentrant=False)
        else:
            streams, valid = self.dgse(original, lengths)
        features = self.rpm(streams, valid)
        speech, reads, attention = self.rcdr(streams, features, valid)
        return dict(
            student_logit=self.classifier(speech).squeeze(-1),
            student_tokens=features,
            student_reads=reads,
            student_attention=attention,
            streams=streams,
            valid=valid,
        )

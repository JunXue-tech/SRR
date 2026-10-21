"""Losses, configuration, checkpoint conversion, and distributed evaluation."""

import argparse
import datetime
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve, roc_auc_score
from load_dataset import RawEvalDataset, PairedEval, DistributedEvalSampler
from distillation import DistillationLoss



class StudentLoss(nn.Module):
    """Supervised student BCE, per example (real=0, fake=1)."""

    def forward(self, result, labels):
        return F.binary_cross_entropy_with_logits(
            result["student_logit"].float(), labels.float(), reduction="none"
        )


class TeacherLoss(nn.Module):
    """Teacher classifier BCE plus SR-only auxiliary BCE, both weighted by 1."""

    def forward(self, result, labels):
        teacher = F.binary_cross_entropy_with_logits(
            result["teacher_logit"].float(), labels.float(), reduction="none"
        )
        sr = F.binary_cross_entropy_with_logits(
            result["superres_logit"].float(), labels.float(), reduction="none"
        )
        return teacher + sr, {"teacher_bce": teacher, "sr_bce": sr}


class TrainingLoss(nn.Module):
    """L = L_student + L_teacher + kd_weight * L_distillation (joint mode)."""

    def __init__(self, mode, kd_weight=0.3, temperature=2.0):
        super().__init__()
        self.mode, self.kd_weight = mode, kd_weight
        self.student = StudentLoss()
        self.teacher = TeacherLoss()
        self.distillation = DistillationLoss(temperature)

    def forward(self, result, labels):
        total = torch.zeros_like(labels, dtype=torch.float32)
        terms = {}
        if self.mode != "teacher":
            loss = self.student(result, labels)
            total = total + loss
            terms["student_loss"] = loss
        if self.mode != "student":
            loss, subterms = self.teacher(result, labels)
            total = total + subterms["teacher_bce"] + subterms["sr_bce"]
            terms.update(subterms)
            terms["teacher_loss"] = loss
        if self.mode == "distill":
            loss, subterms = self.distillation(result)
            total = total + self.kd_weight * loss
            terms.update(subterms)
            terms["distillation_loss"] = loss
        return total, terms


def parser(training=False):
    p = argparse.ArgumentParser(description="FlashSR-guided speech deepfake detection")
    p.add_argument("--xlsr-checkpoint", required=True)
    p.add_argument("--xlsr-output-layer", type=int, default=23, help="23 for standard fairseq; -1 for all-layer fork")
    p.add_argument("--response-stats", default=None)
    p.add_argument(
        "--batch-size", type=int, default=12 if training else 8, help="Per-GPU micro batch"
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--fp32", action="store_true", help="Disable BF16 autocast")
    p.add_argument("--xlsr-checkpointing", action="store_true")
    p.add_argument("--no-spectral-checkpointing", action="store_true")
    if training:
        p.add_argument("--mode", choices=["distill", "teacher", "student"], default="distill")
        p.add_argument("--output-dir", default="runs/distill")
        p.add_argument("--train-manifest", required=True)
        p.add_argument("--dev-manifest", required=True)
        p.add_argument("--dev-sr-manifest", default=None)
        p.add_argument("--epochs", type=int, default=50)
        p.add_argument("--patience", type=int, default=10)
        p.add_argument("--lr", type=float, default=1e-6)
        p.add_argument("--global-batch", type=int, default=60)
        p.add_argument("--weight-decay", type=float, default=0.01)
        p.add_argument("--gradient-clip", type=float, default=1.0)
        p.add_argument("--rawboost-algo", type=int, choices=[4, 5], default=4)
        p.add_argument("--augmentation-probability", type=float, default=0.9)
        p.add_argument("--kd-weight", type=float, default=0.3)
        p.add_argument("--temperature", type=float, default=2.0)
        p.add_argument("--dev-batch-size", type=int, default=0, help="0: student 64, teacher 8")
        p.add_argument("--resume", help="Restore optimizer/main RNG; persistent worker augmentation RNG is not restored")
        p.add_argument(
            "--init-checkpoint", help="Initialize from model weights"
        )
        p.add_argument("--smoke-updates", type=int, default=0)
    else:
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--manifest", required=True)
        p.add_argument("--sr-manifest", default=None)
        p.add_argument("--name", default="eval")
        p.add_argument(
            "--smoke-utterances",
            type=int,
            default=0,
            help="Diagnostic subset only; never reported as full results",
        )
    return p


def model_config(args):
    return {
        "paths": {"xlsr_checkpoint": args.xlsr_checkpoint, "response_stats": args.response_stats},
        "training": {
            "xlsr_gradient_checkpointing": args.xlsr_checkpointing,
            "xlsr_output_layer": None if args.xlsr_output_layer < 0 else args.xlsr_output_layer,
            "spectral_gradient_checkpointing": not args.no_spectral_checkpointing,
        },
    }


def setup(args):
    rank, local, world = [
        int(os.environ.get(k, d)) for k, d in [("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)]
    ]
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return rank, local, world, torch.device("cuda", local)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def atomic_save(path, value):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_json(path, value):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_checkpoint(path):
    """Hash and load the same open inode even if a training job replaces best.pt."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        handle.seek(0)
        value = torch.load(handle, map_location="cpu", weights_only=True)
    return value, digest.hexdigest()


def load_weights(model, checkpoint, mode):
    """Load a pure branch state dictionary or a local training checkpoint strictly."""
    state = checkpoint.get("model", checkpoint)
    if all(isinstance(v, torch.Tensor) for v in state.values()) and any(k.startswith("dgse.") for k in state):
        if mode == "distill":
            raise ValueError("Joint initialization requires a joint checkpoint")
        model.load_state_dict(state, strict=True)
        return
    if mode == "distill":
        model.load_state_dict(state, strict=True)
        return
    selected = {}
    for k, v in state.items():
        if k.startswith("student.dgse."):
            selected["dgse." + k[len("student.dgse."):]] = v
        elif k.startswith(mode + "."):
            selected[k[len(mode) + 1:]] = v
    model.load_state_dict(selected, strict=True)


def make_loader(dataset, args, rank, world, batch_size=None, batch_sampler=None):
    kw = dict(
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers else None,
    )
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **kw)
    return DataLoader(
        dataset,
        batch_size=batch_size or args.batch_size,
        sampler=DistributedEvalSampler(dataset, rank, world),
        **kw,
    )


@torch.inference_mode()
def evaluate(model, loader, branch, args, rank, world, device, destination, name):
    model.eval()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    loss_sum = 0.0
    count = 0
    started = time.time()
    with (destination / f"{name}.rank{rank}.jsonl").open("w") as f:
        for step, batch in enumerate(loader, 1):
            if branch == "teacher":
                raw, n, sr, m, labels, ids, windows, _ = batch
            else:
                raw, n, labels, ids, windows = batch
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
                if branch == "teacher":
                    result = model(raw.to(device), n.to(device), sr.to(device), m.to(device))
                else:
                    result = model(raw.to(device), n.to(device))
            logits = result[branch + "_logit"].float()
            loss_sum += F.binary_cross_entropy_with_logits(
                logits, labels.to(device).float(), reduction="sum"
            ).item()
            count += len(labels)
            if rank == 0 and (step == 1 or step % 250 == 0):
                print(
                    json.dumps(
                        dict(
                            event="eval_progress",
                            dataset=name,
                            branch=branch,
                            local_step=step,
                            utterances_per_second=count / max(1e-6, time.time() - started),
                        )
                    ),
                    flush=True,
                )
            for uid, y, score, w in zip(
                ids, labels.tolist(), logits.cpu().tolist(), windows.tolist()
            ):
                f.write(json.dumps([uid, y, score, w]) + "\n")
    totals = torch.tensor([loss_sum, count], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(totals)
    barrier()
    result = None
    if rank == 0:
        scores = {}
        labels = {}
        seen = set()
        for r in range(world):
            for line in (destination / f"{name}.rank{r}.jsonl").open():
                uid, y, score, w = json.loads(line)
                if uid in scores or w != 0:
                    raise RuntimeError("Expected exactly one first crop per utterance")
                seen.add((uid, w))
                scores[uid] = score
                labels[uid] = y
        ids = sorted(scores)
        y = np.array([labels[k] for k in ids])
        values = np.array([scores[k] for k in ids])
        if len(seen) != int(totals[1]):
            raise RuntimeError("Evaluation count mismatch")
        fpr, tpr, threshold = roc_curve(y, values, pos_label=1)
        fnr = 1 - tpr
        i = int(np.nanargmin(np.abs(fpr - fnr)))
        result = dict(
            eer_percent=float(50 * (fpr[i] + fnr[i])),
            auc=float(roc_auc_score(y, values)),
            eer_threshold=float(threshold[i]),
            loss=float(totals[0] / totals[1]),
            loss_definition=branch + "_utterance_bce",
            utterances=len(ids),
            windows=len(seen),
        )
        with (destination / f"{name}.scores").open("w") as f:
            for uid, label, score in zip(ids, y, values):
                f.write(f"{uid} {label} {score:.9f}\n")
    barrier()
    return result


def evaluation_main(branch):
    args = parser().parse_args()
    rank, local, world, device = setup(args)
    checkpoint, checkpoint_sha256 = read_checkpoint(args.checkpoint)
    if branch == "teacher":
        from teacher import TeacherModel

        model = TeacherModel(model_config(args))
    else:
        from student import StudentModel

        model = StudentModel(model_config(args))
    load_weights(model, checkpoint, branch)
    del checkpoint
    model.to(device)
    out = Path(args.output_dir)
    if args.smoke_utterances:
        out = out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    binding = dict(branch=branch, checkpoint_sha256=checkpoint_sha256,
                   input_samples=64600, crop="first", bf16=not args.fp32,
                   manifest_sha256=file_hash(args.manifest))
    if args.sr_manifest:
        binding["sr_manifest_sha256"] = file_hash(args.sr_manifest)
    if rank == 0:
        path = out / "binding.json"
        if path.exists() and json.loads(path.read_text()) != binding:
            raise ValueError("Use a new output directory for a different checkpoint or protocol")
        atomic_json(path, binding)
    barrier()
    dataset = PairedEval(args.manifest, args.sr_manifest) if branch == "teacher" else RawEvalDataset(args.manifest)
    expected = len(dataset)
    if args.smoke_utterances:
        indices = []
        for label in (0, 1):
            indices.extend([i for i, y in enumerate(dataset.labels) if y == label][:max(1, args.smoke_utterances // 2)])
        dataset = Subset(dataset, indices)
        expected = len(dataset)
    result = evaluate(model, make_loader(dataset, args, rank, world), branch, args,
                      rank, world, device, out, args.name)
    if rank == 0:
        if result["utterances"] != expected or result["windows"] != expected:
            raise RuntimeError("Expected exactly one prediction per utterance")
        result["diagnostic_subset"] = bool(args.smoke_utterances)
        atomic_json(out / "results.json", {args.name: result})
        print(json.dumps({args.name: result}), flush=True)
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()

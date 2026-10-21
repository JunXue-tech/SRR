"""Train a teacher, a student, or the joint FlashSR distillation framework."""

import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from student import StudentModel
from teacher import TeacherModel
from load_dataset import MatchedDataset, RawEvalDataset, PairedEval, GlobalBatchSampler
from utils import (
    parser,
    model_config,
    setup,
    barrier,
    atomic_save,
    atomic_json,
    file_hash,
    load_weights,
    make_loader,
    evaluate as evaluate_dataset,
    TrainingLoss,
)


class DistillationModel(nn.Module):
    """One shared DGSE, two explicit models; teacher speech path is detached."""

    def __init__(self, config):
        super().__init__()
        self.student = StudentModel(config)
        self.teacher = TeacherModel(config, shared_encoder=True)

    def forward(self, original, lengths, superres=None, superres_lengths=None):
        result = self.student(original, lengths)
        if superres is not None:
            result.update(
                self.teacher(
                    original,
                    lengths,
                    superres,
                    superres_lengths,
                    result["streams"].detach(),
                    result["valid"],
                )
            )
        return result


def run_signature(args, world):
    """Resume may extend epochs or change output location, not the experiment."""
    ignored = {
        "epochs",
        "output_dir",
        "resume",
        "init_checkpoint",
        "smoke_updates",
        "dev_batch_size",
        "workers",
    }
    result = {k: v for k, v in vars(args).items() if k not in ignored}
    result["world_size"] = world
    result["train_sha256"] = file_hash(args.train_manifest)
    result["dev_sha256"] = file_hash(args.dev_manifest)
    if args.mode != "student":
        result["response_stats_sha256"] = file_hash(args.response_stats)
    if args.mode == "teacher":
        result["dev_sr_sha256"] = file_hash(args.dev_sr_manifest)
    result["source_sha256"] = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    return result


def build_optimizer(base, args):
    params = [(name, p) for name, p in base.named_parameters() if p.requires_grad]
    decay = [
        p
        for name, p in params
        if p.ndim > 1 and "norm" not in name.lower() and not name.endswith("bias")
    ]
    ids = {id(p) for p in decay}
    other = [p for _, p in params if id(p) not in ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": other, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    return optimizer, params


def train_one_epoch(model, loader, optimizer, criterion, sampler, params, args, rank, world, device, epoch, global_step, log):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    epoch_loss = 0.0
    epoch_count = 0.0
    update_loss = 0.0
    update_count = 0.0
    term_sums = {}
    torch.cuda.reset_peak_memory_stats()
    for step, (raw, n, sr, m, labels, weight) in enumerate(loader, 1):
        sync = step % sampler.accumulation == 0
        with contextlib.nullcontext() if sync or world == 1 else model.no_sync():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
                if args.mode == "student":
                    result = model(raw.to(device, non_blocking=True), n.to(device, non_blocking=True))
                else:
                    result = model(raw.to(device, non_blocking=True), n.to(device, non_blocking=True), sr.to(device, non_blocking=True), m.to(device, non_blocking=True))
            values, terms = criterion(result, labels.to(device, non_blocking=True).float())
            weight = weight.to(device, non_blocking=True).float()
            loss = (values * weight).sum() * world / args.global_batch
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
        if args.smoke_updates and step == 1:
            missing = [name for name, p in params if p.grad is None]
            if missing:
                raise RuntimeError("Missing gradients: " + str(missing))
            log(event="gradient_check", parameters_with_gradient=len(params))
        update_loss += (values.detach() * weight).sum().item()
        update_count += weight.sum().item()
        for key, value in terms.items():
            term_sums[key] = term_sums.get(key, 0.0) + (value.detach() * weight).sum().item()
        del result, values, loss, terms
        if not sync:
            continue
        norm = torch.nn.utils.clip_grad_norm_(
            [p for _, p in params], args.gradient_clip, error_if_nonfinite=True
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        keys = sorted(term_sums)
        totals = torch.tensor(
            [update_loss, update_count] + [term_sums[k] for k in keys],
            device=device,
            dtype=torch.float64,
        )
        if world > 1:
            dist.all_reduce(totals)
        if int(totals[1]) != args.global_batch:
            raise RuntimeError("Global batch changed")
        epoch_loss += float(totals[0])
        epoch_count += args.global_batch
        if global_step <= 4 or global_step % 25 == 0:
            log(
                event="train_step",
                epoch=epoch + 1,
                global_step=global_step,
                loss=float(totals[0] / args.global_batch),
                **{k: float(totals[i + 2] / args.global_batch) for i, k in enumerate(keys)},
                samples_per_second=epoch_count / (time.time() - started),
                gradient_norm=float(norm),
                peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9
            )
        update_loss = 0.0
        update_count = 0.0
        term_sums = {}
        if args.smoke_updates and global_step >= args.smoke_updates:
            break
    return global_step, epoch_loss / epoch_count


def save_branch_checkpoints(base, mode, out, config, epoch, dev, signature):
    """Export self-contained teacher/student weights, each with its own DGSE."""
    state=base.state_dict()
    for branch in (['teacher','student'] if mode=='distill' else [mode]):
        if mode=='distill':
            weights={k[len(branch)+1:]:v for k,v in state.items() if k.startswith(branch+'.')}
            if branch=='teacher':
                weights.update({k[len('student.'):]:v for k,v in state.items() if k.startswith('student.dgse.')})
        else: weights=state
        atomic_save(out/f'best_{branch}.pt', {k: v.detach().cpu().contiguous() for k, v in weights.items()})


def evaluate(base, dev, branch, args, rank, world, device, out, batch_size):
    return evaluate_dataset(base,make_loader(dev,args,rank,world,batch_size=batch_size),branch,args,rank,world,device,out,'dev')


def smoke_evaluate(base, dev, branch, args, rank, world, device, out, batch_size, config, log):
    chosen=[]
    for label in (0,1): chosen.extend(np.flatnonzero(np.asarray(dev.labels)==label)[:4].tolist())
    result=evaluate_dataset(base,make_loader(Subset(dev,chosen),args,rank,world,batch_size=batch_size),branch,args,rank,world,device,out,'smoke_dev')
    if rank==0:
        save_branch_checkpoints(base,args.mode,out,config,0,result,{})
        atomic_json(out/'SMOKE_COMPLETE.json',result)
        log(event='smoke_complete',**result)
    barrier()
    if dist.is_initialized(): dist.destroy_process_group()


def train(args):
    if args.resume and args.init_checkpoint:
        raise ValueError("Choose resume OR weight initialization")
    if args.global_batch < 2 or args.global_batch % 2:
        raise ValueError("Global balanced batch must be positive and even")
    if not 0 <= args.augmentation_probability <= 1:
        raise ValueError("Invalid augmentation probability")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("Invalid training limits")
    if args.mode != "student" and not args.response_stats:
        raise ValueError("Teacher training requires --response-stats estimated from training data")
    rank, local, world, device = setup(args)
    os.environ["RAWBOOST_ALGO"] = str(args.rawboost_algo)
    os.environ["RAWBOOST_PROBABILITY"] = str(args.augmentation_probability)
    out = Path(args.output_dir)
    if args.smoke_updates:
        out = out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and ((out / "resume.pt").exists() or (out / "best_student.pt").exists() or (out / "best_teacher.pt").exists()):
        raise ValueError("Existing checkpoints: supply --resume or a fresh output directory")

    def log(**record):
        if rank == 0:
            record["time"] = time.time()
            line = json.dumps(record)
            print(line, flush=True)
            with (out / "metrics.jsonl").open("a") as f:
                f.write(line + "\n")

    config = model_config(args)
    constructor = {"distill": DistillationModel, "teacher": TeacherModel, "student": StudentModel}[
        args.mode
    ]
    base = constructor(config).to(device, non_blocking=True)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        load_weights(base, ckpt, args.mode)
        del ckpt
    train = MatchedDataset(args.train_manifest, needs_sr=args.mode != "student")
    branch = "teacher" if args.mode == "teacher" else "student"
    dev = (
        PairedEval(args.dev_manifest, args.dev_sr_manifest)
        if branch == "teacher"
        else RawEvalDataset(args.dev_manifest)
    )
    sampler = GlobalBatchSampler(
        train.labels, rank, world, args.batch_size, args.seed, args.global_batch
    )
    dev_batch = args.dev_batch_size or (8 if branch == "teacher" else 64)
    # Workers persist across epochs.
    train_loader = make_loader(train, args, rank, world, batch_sampler=sampler)
    dev_loader = make_loader(dev, args, rank, world, batch_size=dev_batch)
    optimizer, params = build_optimizer(base, args)
    criterion = TrainingLoss(args.mode, args.kd_weight, args.temperature)
    signature = run_signature(args, world)
    epoch_start = 0
    global_step = 0
    best_loss = float("inf")
    best_eer = float("inf")
    bad_epochs = 0
    if args.resume:
        log(event="resume_note", message="Persistent worker RNG is not checkpointed; resumed augmentation draws are not bitwise identical to uninterrupted training.")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt.get("signature") != signature:
            raise ValueError(
                "Resume configuration/source/data/world-size mismatch; use --init-checkpoint for weights only"
            )
        base.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        epoch_start = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_loss = ckpt["best_dev_loss"]
        best_eer = ckpt["best_dev_eer"]
        bad_epochs = ckpt["bad_epochs"]
        rng = ckpt["rng_states"][rank]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        del ckpt
    model = (
        DDP(
            base,
            device_ids=[local],
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        if world > 1
        else base
    )
    if rank == 0:
        atomic_json(
            out / "config.json",
            dict(arguments=vars(args), signature=signature, model_config=config),
        )
    log(
        event="train_start",
        mode=args.mode,
        world_size=world,
        micro_batch=args.batch_size,
        global_batch=args.global_batch,
        accumulation=sampler.accumulation,
        start_epoch=epoch_start,
        input_samples=64600,
        sr_samples=193800,
    )
    for epoch in range(epoch_start, args.epochs):
        if bad_epochs >= args.patience:
            break
        sampler.set_epoch(epoch)
        started = time.time()
        global_step, train_loss = train_one_epoch(model, train_loader, optimizer, criterion, sampler, params, args, rank, world, device, epoch, global_step, log)
        if args.smoke_updates:
            smoke_evaluate(base, dev, branch, args, rank, world, device, out, dev_batch, config, log)
            return
        result = evaluate_dataset(base, dev_loader, branch, args, rank, world, device, out, "dev")
        if rank == 0:
            improved = result["loss"] < best_loss
            best_loss = min(best_loss, result["loss"])
            best_eer = min(best_eer, result["eer_percent"])
            bad_epochs = 0 if improved else bad_epochs + 1
            if improved:
                save_branch_checkpoints(base, args.mode, out, config, epoch + 1, result, signature)
                atomic_json(out / "best_loss.json", dict(epoch=epoch + 1, mode=args.mode, **result))
            log(
                event="epoch",
                epoch=epoch + 1,
                train_loss=train_loss,
                bad_epochs=bad_epochs,
                best_dev_loss=best_loss,
                seconds=time.time() - started,
                **result
            )
        values = torch.tensor([best_loss, best_eer, bad_epochs], dtype=torch.float64, device=device)
        if world > 1:
            dist.broadcast(values, src=0)
        best_loss, best_eer, bad_epochs = float(values[0]), float(values[1]), int(values[2])
        rng = dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state(device),
        )
        rng_states = [None] * world
        if world > 1:
            dist.all_gather_object(rng_states, rng)
        else:
            rng_states[0] = rng
        if rank == 0:
            atomic_save(
                out / "resume.pt",
                dict(
                    model=base.state_dict(),
                    optimizer=optimizer.state_dict(),
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_dev_loss=best_loss,
                    best_dev_eer=best_eer,
                    bad_epochs=bad_epochs,
                    rng_states=rng_states,
                    mode=args.mode,
                    config=config,
                    signature=signature,
                ),
            )
        barrier()
    if rank == 0:
        log(event="training_complete", best_dev_loss=best_loss, best_dev_eer=best_eer)
        atomic_json(out / "TRAINING_COMPLETE.json", dict(best_dev_loss=best_loss, epoch=epoch+1))
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parser(training=True).parse_args()
    train(args)


if __name__ == "__main__":
    main()

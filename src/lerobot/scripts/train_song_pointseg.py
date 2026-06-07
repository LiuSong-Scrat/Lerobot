#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/lerobot_hf_datasets_cache")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.song_pointseg import (
    DEFAULT_FUTURE_OFFSETS,
    EMATeacher,
    PseudoLabelConfig,
    SongPointSegCachedDataset,
    SongPointSegLoss,
    SongPointSegLossConfig,
    SongPointSegNet,
    SongTemporalPointCloudDataset,
    force_small_current_clouds_foreground,
    generate_pseudo_labels,
    move_batch_to_device,
    parse_future_offsets,
    pretty_metrics,
    refine_pseudo_labels_with_teacher,
    save_pointseg_config,
    save_pointseg_npz,
    song_pointseg_collate,
    write_role_ply,
)
from lerobot.utils.random_utils import set_seed

DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "SONG_POINTSEG_DATASET",
        "/home/liusong/ProgramFiles/BestMan/Dataset/dataset/test3/src_hdf5_to_lerobot/lerobot_datasets/temp",
    )
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "SONG_POINTSEG_OUTPUT_DIR",
        "/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/song_pointseg",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unsupervised Song manipulation point-cloud segmentation.")
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset.root", dest="dataset_root", type=str, default=None)
    parser.add_argument("--point-cloud-dir", type=Path, default=None)
    parser.add_argument("--sample-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--future-offsets", type=parse_future_offsets, default=DEFAULT_FUTURE_OFFSETS)
    parser.add_argument("--current-points", type=int, default=50000)
    parser.add_argument("--future-points", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--backbone-type", choices=["litept", "mlp"], default="litept")
    parser.add_argument("--grid-size", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--ema-start-step", type=int, default=200)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--vis-freq", type=int, default=500)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _make_lerobot_dataset(args: argparse.Namespace) -> LeRobotDataset:
    repo_id = args.dataset_repo_id
    root = Path(args.dataset_root) if args.dataset_root else None
    max_offset = max(args.future_offsets)
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    fps = int(metadata.fps)
    return LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps={
            "action": [i / fps for i in range(max_offset + 1)],
            "observation.state": [0.0],
        },
    )


def make_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    if args.sample_cache_dir is not None:
        return SongPointSegCachedDataset(args.sample_cache_dir)

    dataset = _make_lerobot_dataset(args)
    dataset_root = Path(getattr(dataset, "root", args.dataset_repo_id))
    point_cloud_dir = args.point_cloud_dir or dataset_root / "point_clouds"
    return SongTemporalPointCloudDataset(
        dataset,
        point_cloud_dir=point_cloud_dir,
        future_offsets=args.future_offsets,
        current_points=args.current_points,
        future_points=args.future_points,
        seed=args.seed,
    )


def pseudo_from_cached_batch(batch: dict) -> dict[str, torch.Tensor]:
    return {
        "priors": batch["pointseg.priors"],
        "labels": batch["pointseg.labels"],
        "weights": batch["pointseg.weights"],
        "class_scores": batch["pointseg.class_scores"],
        "role_scores": batch["pointseg.role_scores"],
        "foreground_score": batch["pointseg.foreground_score"],
    }


def save_visualization(
    output_dir: Path,
    step: int,
    batch: dict,
    outputs: dict,
    pseudo: dict,
    batch_index: int = 0,
) -> None:
    vis_dir = output_dir / "visualizations"
    current_pc = batch["observation.point_cloud"][batch_index].detach().cpu()
    probs = outputs["role_probs"][batch_index].detach().cpu()
    pred_labels = probs.argmax(dim=-1).numpy()
    operation_prob = outputs["operation_prob"][batch_index].detach().cpu().numpy()
    pseudo_labels = pseudo["labels"][batch_index].detach().cpu().numpy()

    write_role_ply(vis_dir / f"step_{step:06d}_pred.ply", current_pc.numpy(), pred_labels, operation_prob)
    write_role_ply(vis_dir / f"step_{step:06d}_pseudo.ply", current_pc.numpy(), pseudo_labels)
    save_pointseg_npz(vis_dir / f"step_{step:06d}.npz", current_pc, {k: v[batch_index] for k, v in outputs.items() if torch.is_tensor(v)}, {k: v[batch_index] for k, v in pseudo.items() if torch.is_tensor(v)})


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: SongPointSegNet,
    optimizer: torch.optim.Optimizer,
    teacher: EMATeacher | None,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    if teacher is not None:
        payload["teacher"] = teacher.model.state_dict()
    torch.save(payload, checkpoint_dir / f"step_{step:06d}.pt")
    torch.save(payload, checkpoint_dir / "last.pt")


def train(args: argparse.Namespace) -> None:
    if args.smoke_test:
        args.steps = min(args.steps, 2)
        args.current_points = min(args.current_points, 512)
        args.future_points = min(args.future_points, 1024)
        args.batch_size = min(args.batch_size, 2)
        args.vis_freq = 1
        args.save_freq = 2

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pseudo_cfg = PseudoLabelConfig()
    loss_cfg = SongPointSegLossConfig()
    save_pointseg_config(args.output_dir / "pointseg_config.json", args, pseudo_cfg, loss_cfg)

    dataset = make_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=song_pointseg_collate,
    )
    iterator = iter(dataloader)

    model = SongPointSegNet(backbone_type=args.backbone_type, grid_size=args.grid_size).to(device)
    criterion = SongPointSegLoss(loss_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    teacher: EMATeacher | None = None

    progress = tqdm(range(1, args.steps + 1), desc="Song pointseg", unit="step")
    for step in progress:
        start_time = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        batch = move_batch_to_device(batch, device)
        current_pc = batch["observation.point_cloud"]
        current_is_pad = batch.get("observation.point_cloud_is_pad")
        uses_cached_pseudo = "pointseg.priors" in batch

        with torch.no_grad():
            if uses_cached_pseudo:
                pseudo = pseudo_from_cached_batch(batch)
            else:
                pseudo = generate_pseudo_labels(
                    current_pc,
                    batch["observation.point_cloud_future"],
                    batch["future_ee_poses"],
                    batch["future_is_pad"],
                    current_is_pad=current_is_pad,
                    future_point_is_pad=batch.get("observation.point_cloud_future_is_pad"),
                    config=pseudo_cfg,
                )
                pseudo = force_small_current_clouds_foreground(
                    pseudo,
                    current_pc,
                    args.current_points,
                    current_is_pad,
                )
            if current_is_pad is not None:
                pseudo["point_is_pad"] = current_is_pad
            if teacher is not None:
                teacher_outputs = teacher.model(current_pc, priors=pseudo["priors"], point_is_pad=current_is_pad)
                pseudo = refine_pseudo_labels_with_teacher(pseudo, teacher_outputs["role_logits"], config=pseudo_cfg)

        model.train()
        outputs = model(current_pc, priors=pseudo["priors"], point_is_pad=current_is_pad)
        loss, metrics = criterion(outputs, pseudo, current_pc)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        if teacher is None and not args.no_ema and step >= args.ema_start_step:
            teacher = EMATeacher(model, decay=args.ema_decay)
        elif teacher is not None:
            teacher.update(model)

        metrics_for_log = {**metrics, "step_s": time.perf_counter() - start_time}
        progress.set_postfix(
            {
                key: f"{float(value.item()) if torch.is_tensor(value) else value:.3f}"
                for key, value in metrics_for_log.items()
                if key in {"loss", "pseudo_foreground_ratio", "pred_foreground_ratio", "step_s"}
            }
        )

        if args.log_freq > 0 and step % args.log_freq == 0:
            print(pretty_metrics(metrics_for_log, step))

        if args.vis_freq > 0 and step % args.vis_freq == 0:
            with torch.no_grad():
                save_visualization(args.output_dir, step, batch, outputs, pseudo)

        if args.save_freq > 0 and step % args.save_freq == 0:
            save_checkpoint(args.output_dir, step, model, optimizer, teacher, args)

    save_checkpoint(args.output_dir, args.steps, model, optimizer, teacher, args)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

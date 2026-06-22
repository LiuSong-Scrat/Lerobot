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
        "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/real_lerobot_dataset",
    )
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "SONG_POINTSEG_OUTPUT_DIR",
        "/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/song_pointseg",
    )
)
import math
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack
def create_frame(position, rot_matrix, scale=0.03):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=scale,
        origin=[0, 0, 0]
    )
    frame.rotate(rot_matrix, center=np.zeros(3))
    frame.translate(position)
    return frame
def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns


def _skew(vec: Tensor) -> Tensor:
    x, y, z = vec.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def _eye4_like(shape: torch.Size | tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    eye = torch.eye(4, device=device, dtype=dtype)
    return eye.expand(*shape, 4, 4).clone()


def so3_exp(omega: Tensor, eps: float = 1e-6) -> Tensor:
    omega = omega.to(dtype=torch.float32)
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    k = _skew(omega)
    k2 = k @ k
    small = theta < eps
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta.clamp_min(eps),
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(eps),
    )
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(*omega.shape[:-1], 3, 3)
    return eye + a.unsqueeze(-1) * k + b.unsqueeze(-1) * k2


def so3_log(rot: Tensor, eps: float = 1e-6) -> Tensor:
    rot = rot.to(dtype=torch.float32)
    trace = rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    vee = torch.stack(
        [
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * torch.linalg.norm(vee, dim=-1)
    theta = torch.atan2(sine, cosine)
    theta2 = theta * theta
    factor = torch.where(
        sine > eps,
        theta / (2.0 * sine.clamp_min(eps)),
        0.5 + theta2 / 12.0 + theta2 * theta2 / 720.0,
    )
    return factor.unsqueeze(-1) * vee


def se3_exp(xi: Tensor, eps: float = 1e-6) -> Tensor:
    xi = xi.to(dtype=torch.float32)
    v = xi[..., :3]
    omega = xi[..., 3:6]
    rot = so3_exp(omega, eps=eps)
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    k = _skew(omega)
    k2 = k @ k
    small = theta < eps
    a = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(eps),
    )
    b = torch.where(
        small,
        1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0,
        (theta - torch.sin(theta)) / (theta2 * theta).clamp_min(eps),
    )
    eye3 = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(*xi.shape[:-1], 3, 3)
    v_matrix = eye3 + a.unsqueeze(-1) * k + b.unsqueeze(-1) * k2
    trans = (v_matrix @ v.unsqueeze(-1)).squeeze(-1)
    out = _eye4_like(xi.shape[:-1], device=xi.device, dtype=xi.dtype)
    out[..., :3, :3] = rot
    out[..., :3, 3] = trans
    return out


def se3_log(transform: Tensor, eps: float = 1e-6) -> Tensor:
    transform = transform.to(dtype=torch.float32)
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    omega = so3_log(rot, eps=eps)
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    k = _skew(omega)
    k2 = k @ k
    small = theta < eps
    half_theta = 0.5 * theta
    c = torch.where(
        small,
        1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0,
        (1.0 / theta2.clamp_min(eps))
        - (1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta).clamp_min(eps)),
    )
    eye3 = torch.eye(3, device=transform.device, dtype=transform.dtype).expand(*transform.shape[:-2], 3, 3)
    v_inv = eye3 - 0.5 * k + c.unsqueeze(-1) * k2
    v = (v_inv @ trans.unsqueeze(-1)).squeeze(-1)
    # `half_theta` is kept to make the small-angle branch explicit and silence over-eager simplifiers.
    _ = half_theta
    return torch.cat([v, omega], dim=-1)


def se3_left_apply(delta_xi: Tensor, transform: Tensor) -> Tensor:
    return se3_exp(delta_xi) @ transform


def se3_geodesic_loss(pred: Tensor, target: Tensor, trans_weight: float = 1.0, rot_weight: float = 1.0) -> Tensor:
    trans = F.smooth_l1_loss(pred[..., :3, 3], target[..., :3, 3], reduction="none").sum(dim=-1)
    rot = _rotation_geodesic(pred[..., :3, :3], target[..., :3, :3])
    return trans_weight * trans + rot_weight * rot


def _transform_point_cloud_xyzrgb(point_cloud: Tensor, transform: Tensor) -> Tensor:
    xyz = point_cloud[..., :3].to(dtype=torch.float32)
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    xyz_out = torch.matmul(xyz, rot.transpose(-1, -2)) + trans.unsqueeze(-2)
    return torch.cat([xyz_out, point_cloud[..., 3:6].to(dtype=torch.float32)], dim=-1)


def _sample_random_se3(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    trans_scale: float = 0.20,
    rot_scale: float = 0.75,
) -> Tensor:
    xi = torch.randn(batch_size, 6, device=device, dtype=dtype)
    xi[..., :3] = xi[..., :3] * float(trans_scale)
    xi[..., 3:6] = xi[..., 3:6] * float(rot_scale)
    return se3_exp(xi)


def _to_numpy_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _time_gradient(num_steps: int) -> np.ndarray:
    if num_steps <= 1:
        return np.array([[0.1, 0.75, 0.25]], dtype=np.float64)
    t = np.linspace(0.0, 1.0, num_steps, dtype=np.float64)[:, None]
    start = np.array([0.05, 0.55, 1.0], dtype=np.float64)
    middle = np.array([0.10, 0.85, 0.25], dtype=np.float64)
    end = np.array([1.0, 0.18, 0.05], dtype=np.float64)
    first_half = (1.0 - 2.0 * t) * start + (2.0 * t) * middle
    second_half = (2.0 - 2.0 * t) * middle + (2.0 * t - 1.0) * end
    return np.where(t <= 0.5, first_half, second_half).clip(0.0, 1.0)


def _make_sphere(center: np.ndarray, radius: float, color: np.ndarray) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    sphere.translate(center)
    sphere.paint_uniform_color(color.tolist())
    return sphere


def _make_trajectory_lines(positions: np.ndarray, colors: np.ndarray) -> o3d.geometry.LineSet | None:
    if positions.shape[0] < 2:
        return None
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    line_set.lines = o3d.utility.Vector2iVector([[idx, idx + 1] for idx in range(positions.shape[0] - 1)])
    line_colors = 0.5 * (colors[:-1] + colors[1:])
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set


def vis_umi_data(
    action,
    pointcloud,
    *,
    frame_stride: int | None = None,
    max_frames: int = 12,
    frame_scale: float = 0.035,
    point_radius: float = 0.008,
):
    """Visualize a UMI pose9 trajectory with explicit temporal order.

    The trajectory is colored from blue/green at the beginning to red at the end.
    Coordinate frames are drawn sparsely so dense chunks remain readable.
    """
    actions = _to_numpy_array(action).astype(np.float32, copy=False)
    while actions.ndim > 2 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[-1] < 9:
        raise ValueError(f"Expected action shape (T, >=9), got {actions.shape}.")

    cloud = _to_numpy_array(pointcloud).astype(np.float32, copy=False)
    while cloud.ndim > 2 and cloud.shape[0] == 1:
        cloud = cloud[0]
    if cloud.ndim != 2 or cloud.shape[-1] < 3:
        raise ValueError(f"Expected pointcloud shape (N, >=3), got {cloud.shape}.")

    positions = actions[:, :3]
    colors = _time_gradient(positions.shape[0])
    if frame_stride is None:
        frame_stride = max(1, int(math.ceil(positions.shape[0] / max(1, max_frames))))
    frame_indices = list(range(0, positions.shape[0], max(1, int(frame_stride))))
    if positions.shape[0] - 1 not in frame_indices:
        frame_indices.append(positions.shape[0] - 1)

    geometries = [create_frame(np.array([0.0, 0.0, 0.0]), np.eye(3), scale=frame_scale * 1.2)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    if cloud.shape[-1] >= 6:
        rgb = np.clip(cloud[:, 3:6] / 255.0, 0.0, 1.0)
    else:
        rgb = np.full((cloud.shape[0], 3), 0.55, dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    geometries.append(pcd)

    trajectory_lines = _make_trajectory_lines(positions, colors)
    if trajectory_lines is not None:
        geometries.append(trajectory_lines)

    # Small colored beads make the time direction visible even when frames overlap.
    for idx, (position, color) in enumerate(zip(positions, colors, strict=True)):
        radius = point_radius * (1.6 if idx in (0, positions.shape[0] - 1) else 1.0)
        geometries.append(_make_sphere(position, radius, color))

    for idx in frame_indices:
        rot6d = torch.as_tensor(actions[idx, 3:9], dtype=torch.float32)
        rotmat = rot6d_to_matrix(rot6d).cpu().numpy()
        scale = frame_scale * (1.35 if idx in (0, positions.shape[0] - 1) else 1.0)
        geometries.append(create_frame(positions[idx], rotmat, scale=scale))

    print(
        f"Visualizing {positions.shape[0]} poses: blue/green=start, red=end, "
        f"frames={frame_indices}, start={positions[0].round(4)}, end={positions[-1].round(4)}"
    )
    o3d.visualization.draw_geometries(
        geometries,
        window_name="UMI trajectory: blue/green=start, red=end",
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
    parser.add_argument("--future-points", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=8)
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



# idx = 0
# valid_mask = (batch["observation.point_cloud_is_pad"]).sum(1)==0
# for idx,label in enumerate(valid_mask):
#     if label==True:
#         batch_idx = idx

#         point_valid = ~batch["observation.point_cloud_is_pad"][batch_idx]
#         step_valid = ~batch["future_is_pad"][batch_idx]

#         point_cloud = batch["observation.point_cloud"][batch_idx][point_valid]
#         rgb = point_cloud[...,3:]
#         rgb[pseudo['labels'][batch_idx].cpu().numpy()!=1] = 0
#         point_cloud[...,3:] = rgb


#         trajectory = batch["future_ee_poses"][batch_idx][step_valid]

#         print("episode:", batch["episode_index"][batch_idx].item())
#         print("frame:", batch["frame_index"][batch_idx].item())
#         print("raw action start:", batch["action"][batch_idx, 0, :9])
#         print("relative start:", trajectory[0])

#         vis_umi_data(
#             trajectory.cpu().numpy(),
#             point_cloud.cpu().numpy(),
#         )
#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter HDF5 episodes by end-effector trajectory continuity.")
    parser.add_argument("hdf5_dir", nargs="?", default="/home/liusong/temp/temp_with_gripper")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pose-key", default="observations/pose_eular")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mode", choices=("copy", "move", "list"), default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-jumps", type=int, default=5)
    return parser.parse_args()


def iter_hdf5_files(root_dir: str | Path, recursive: bool = True):
    root = Path(root_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"HDF5 folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")
    pattern = "**/*.hdf5" if recursive else "*.hdf5"
    yield from sorted(root.glob(pattern))


def load_positions(hdf5_path: Path, pose_key: str) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as h5_file:
        if pose_key not in h5_file:
            raise KeyError(f"Missing dataset: {pose_key}")
        poses = np.asarray(h5_file[pose_key])
    if poses.ndim != 2 or poses.shape[1] < 3:
        raise ValueError(f"{pose_key} must have shape [frames, >=3], got {poses.shape}")
    return poses[:, :3].astype(np.float64)


def detect_jumps(positions: np.ndarray, threshold: float):
    nonfinite_frames = np.where(~np.isfinite(positions).all(axis=1))[0]
    if len(positions) < 2:
        return np.array([], dtype=int), np.array([], dtype=np.float64), nonfinite_frames
    step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    jump_indices = np.where(~np.isfinite(step_distances) | (step_distances > threshold))[0]
    return jump_indices, step_distances, nonfinite_frames


def is_continuous(hdf5_path: Path, pose_key: str, threshold: float):
    positions = load_positions(hdf5_path, pose_key)
    jump_indices, step_distances, nonfinite_frames = detect_jumps(positions, threshold)
    return len(jump_indices) == 0 and len(nonfinite_frames) == 0, jump_indices, step_distances, nonfinite_frames


def transfer_file(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if mode == "list":
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output file exists: {dst}")
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    args = parse_args()
    src_root = Path(args.hdf5_dir).expanduser().resolve()
    if args.mode != "list" and args.output_dir is None:
        raise ValueError("--output-dir is required for --mode copy or --mode move.")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else None

    good_files: list[Path] = []
    bad_files: list[tuple[Path, str]] = []
    skipped_files: list[tuple[Path, Exception]] = []

    for hdf5_path in iter_hdf5_files(src_root, args.recursive):
        try:
            ok, jump_indices, step_distances, nonfinite_frames = is_continuous(
                hdf5_path,
                args.pose_key,
                args.threshold,
            )
        except Exception as exc:
            skipped_files.append((hdf5_path, exc))
            continue

        if ok:
            good_files.append(hdf5_path)
            if output_dir is not None:
                transfer_file(hdf5_path, output_dir / hdf5_path.relative_to(src_root), args.mode, args.overwrite)
            continue

        details = []
        if len(nonfinite_frames) > 0:
            frames = ", ".join(str(int(i)) for i in nonfinite_frames[: args.max_jumps])
            details.append(f"nonfinite={frames}")
        if len(jump_indices) > 0:
            jumps = []
            for jump_idx in jump_indices[: args.max_jumps]:
                jumps.append(f"{int(jump_idx)}->{int(jump_idx + 1)}:{float(step_distances[jump_idx]):.6f}")
            details.append("jumps=" + ",".join(jumps))
        bad_files.append((hdf5_path, "; ".join(details)))

    print(f"Scanned: {src_root}")
    print(f"Good continuous files: {len(good_files)}")
    print(f"Bad discontinuous files: {len(bad_files)}")
    print(f"Skipped files: {len(skipped_files)}")
    if output_dir is not None and args.mode != "list":
        print(f"Wrote clean files to: {output_dir}")

    if bad_files:
        print("\nBad files:")
        for path, details in bad_files:
            print(f"  {path.name}: {details}")
    if skipped_files:
        print("\nSkipped files:")
        for path, exc in skipped_files:
            print(f"  {path.name}: {exc}")


if __name__ == "__main__":
    main()

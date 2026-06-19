#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

if __package__ and __package__.startswith("benchmarks."):
    from ._paths import REAL_DATA_ROOT
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import REAL_DATA_ROOT


DEFAULT_HDF5_DIR = REAL_DATA_ROOT / "humanhand_offline_demo"
DEFAULT_POSE_KEY = "observations/pose_eular"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Traverse a folder of hdf5 files and print files whose "
            "trajectory has discontinuous position jumps."
        )
    )
    parser.add_argument(
        "--hdf5_dir",
        nargs="?",
        default=DEFAULT_HDF5_DIR,
        help=f"HDF5 folder to scan. Default: {DEFAULT_HDF5_DIR}",
    )
    parser.add_argument(
        "--pose-key",
        default=DEFAULT_POSE_KEY,
        help=f"HDF5 dataset containing pose data. Default: {DEFAULT_POSE_KEY}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help=(
            "Absolute jump threshold in the same unit as pose xyz. "
            "Default: 0.1"
        ),
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan subdirectories recursively. Default: true",
    )
    parser.add_argument(
        "--names-only",
        action="store_true",
        help="Only print discontinuous hdf5 filenames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional destination for continuous files. Omit it for report-only mode.",
    )
    parser.add_argument("--mode", choices=("copy", "move"), default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-jumps",
        type=int,
        default=5,
        help="Maximum jump details to print per file. Default: 5",
    )
    return parser.parse_args()


def iter_hdf5_files(root_dir, recursive=True):
    root = Path(root_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"HDF5 folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    pattern = "**/*.hdf5" if recursive else "*.hdf5"
    yield from sorted(root.glob(pattern))


def load_positions(hdf5_path, pose_key):
    with h5py.File(hdf5_path, "r") as f:
        if pose_key not in f:
            raise KeyError(f"Missing dataset: {pose_key}")
        poses = np.asarray(f[pose_key])

    if poses.ndim != 2 or poses.shape[1] < 3:
        raise ValueError(
            f"{pose_key} must have shape [frames, >=3], got {poses.shape}"
        )
    return poses[:, :3].astype(np.float64)


def detect_jumps(positions, threshold):
    nonfinite_frames = np.where(~np.isfinite(positions).all(axis=1))[0]
    if len(positions) < 2:
        return np.array([], dtype=int), np.array([], dtype=np.float64), nonfinite_frames

    step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    jump_indices = np.where(~np.isfinite(step_distances) | (step_distances > threshold))[0]
    return jump_indices, step_distances, nonfinite_frames


def transfer_continuous_file(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output file exists: {dst}")
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main():
    args = parse_args()
    source_root = Path(args.hdf5_dir).expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve() if args.output_dir is not None else None
    good_files = []
    bad_files = []
    skipped_files = []

    for hdf5_path in iter_hdf5_files(source_root, args.recursive):
        try:
            positions = load_positions(hdf5_path, args.pose_key)
            print(hdf5_path," len = ",positions.shape[0])
            jump_indices, step_distances, nonfinite_frames = detect_jumps(
                positions, args.threshold
            )
        except Exception as exc:
            skipped_files.append((hdf5_path, exc))
            continue

        if len(jump_indices) > 0 or len(nonfinite_frames) > 0:
            bad_files.append(
                {
                    "path": hdf5_path,
                    "positions": positions,
                    "jump_indices": jump_indices,
                    "step_distances": step_distances,
                    "nonfinite_frames": nonfinite_frames,
                }
            )
            continue

        good_files.append(hdf5_path)
        if output_root is not None:
            transfer_continuous_file(
                hdf5_path,
                output_root / hdf5_path.relative_to(source_root),
                args.mode,
                args.overwrite,
            )

    if args.names_only:
        for item in bad_files:
            print(item["path"].name)
        return

    print(f"Scanned folder: {source_root}")
    print(f"Pose dataset: {args.pose_key}")
    print(f"Jump threshold: {args.threshold}")
    print(f"Continuous hdf5 count: {len(good_files)}")
    print(f"Discontinuous hdf5 count: {len(bad_files)}")
    if output_root is not None:
        print(f"Continuous files written to: {output_root}")
    print()

    if bad_files:
        print("Discontinuous hdf5 files:")
        for item in bad_files:
            print(item["path"].name)
        print()

    for item in bad_files:
        path = item["path"]
        positions = item["positions"]
        jump_indices = item["jump_indices"]
        step_distances = item["step_distances"]
        nonfinite_frames = item["nonfinite_frames"]

        max_step = float(np.nanmax(step_distances)) if step_distances.size else 0.0
        print(
            f"{path.name}: frames={len(positions)}, "
            f"jumps={len(jump_indices)}, max_step={max_step:.6f}"
        )

        if len(nonfinite_frames) > 0:
            frames = ", ".join(str(int(i)) for i in nonfinite_frames[: args.max_jumps])
            suffix = " ..." if len(nonfinite_frames) > args.max_jumps else ""
            print(f"  non-finite position frames: {frames}{suffix}")

        for jump_idx in jump_indices[: args.max_jumps]:
            start = positions[jump_idx]
            end = positions[jump_idx + 1]
            print(
                f"  frame {jump_idx}->{jump_idx + 1}: "
                f"step={step_distances[jump_idx]:.6f}, "
                f"from={np.array2string(start, precision=6)}, "
                f"to={np.array2string(end, precision=6)}"
            )
        if len(jump_indices) > args.max_jumps:
            print(f"  ... {len(jump_indices) - args.max_jumps} more jumps")

    if skipped_files:
        print()
        print(f"Skipped hdf5 files: {len(skipped_files)}")
        for path, exc in skipped_files:
            print(f"  {path.name}: {exc}")


if __name__ == "__main__":
    main()

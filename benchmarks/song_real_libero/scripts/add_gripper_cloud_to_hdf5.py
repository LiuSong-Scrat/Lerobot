import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

if __package__ and __package__.startswith("benchmarks."):
    from ._paths import REAL_DATA_ROOT
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import REAL_DATA_ROOT


DEFAULT_INPUT_DIR = REAL_DATA_ROOT / "temp/hdf5_without_gripper"
DEFAULT_OUTPUT_DIR = REAL_DATA_ROOT / "temp/hdf5_with_gripper"


def natural_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def normalize_widths(eff_angular, already_normalized=False):
    widths = np.asarray(eff_angular, dtype=np.float32).reshape(-1).copy()
    if already_normalized:
        return np.clip(widths, 0.0, 1.0)

    width_range = widths.max() - widths.min()
    if width_range != 0:
        widths = (widths - widths.min()) / width_range
    else:
        widths[widths >= 0] = 1.0
    return np.clip(widths, 0.0, 1.0)


def allocate_counts(total, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if total <= 0:
        return np.zeros(len(weights), dtype=np.int64)
    if weights.sum() <= 0:
        counts = np.zeros(len(weights), dtype=np.int64)
        counts[:total] = 1
        return counts

    expected = total * weights / weights.sum()
    counts = np.floor(expected).astype(np.int64)
    remainder = total - counts.sum()
    if remainder > 0:
        order = np.argsort(expected - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


def box_faces(min_corner, size):
    sx, sy, sz = size
    x0, y0, z0 = min_corner
    x1, y1, z1 = min_corner + size
    return [
        (sy * sz, np.array([x0, y0, z0]), np.array([0.0, sy, 0.0]), np.array([0.0, 0.0, sz])),
        (sy * sz, np.array([x1, y0, z0]), np.array([0.0, sy, 0.0]), np.array([0.0, 0.0, sz])),
        (sx * sz, np.array([x0, y0, z0]), np.array([sx, 0.0, 0.0]), np.array([0.0, 0.0, sz])),
        (sx * sz, np.array([x0, y1, z0]), np.array([sx, 0.0, 0.0]), np.array([0.0, 0.0, sz])),
        (sx * sy, np.array([x0, y0, z0]), np.array([sx, 0.0, 0.0]), np.array([0.0, sy, 0.0])),
        (sx * sy, np.array([x0, y0, z1]), np.array([sx, 0.0, 0.0]), np.array([0.0, sy, 0.0])),
    ]


def sample_box_surface(min_corner, size, count, rng):
    faces = box_faces(np.asarray(min_corner, dtype=np.float64), np.asarray(size, dtype=np.float64))
    counts = allocate_counts(count, [face[0] for face in faces])
    samples = []
    for face_count, (_, origin, axis_a, axis_b) in zip(counts, faces):
        if face_count == 0:
            continue
        uv = rng.random((face_count, 2), dtype=np.float64)
        samples.append(origin + uv[:, :1] * axis_a + uv[:, 1:] * axis_b)
    if not samples:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(samples)


def create_gripper_points(
    width_percent,
    pose,
    count,
    rng,
    gripper_len=0.06,
    max_width=0.06,
    finger_length=0.08,
    finger_thickness=0.01,
    base_thickness=0.01,
    handle_length=0.05,
):
    width = float(width_percent) * max_width
    boxes = [
        (
            np.array([width / 2.0, base_thickness, 0.0]),
            np.array([finger_thickness, finger_length, finger_thickness]),
        ),
        (
            np.array([-width / 2.0 - finger_thickness, base_thickness, 0.0]),
            np.array([finger_thickness, finger_length, finger_thickness]),
        ),
        (
            np.array([-max_width / 2.0 - finger_thickness / 2.0, 0.0, 0.0]),
            np.array([max_width + finger_thickness, base_thickness, finger_thickness]),
        ),
        (
            np.array([-base_thickness / 2.0, -handle_length, 0.0]),
            np.array([base_thickness, handle_length, base_thickness]),
        ),
    ]

    box_areas = [2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2]) for _, size in boxes]
    box_counts = allocate_counts(count, box_areas)
    points = []
    for box_count, (min_corner, size) in zip(box_counts, boxes):
        points.append(sample_box_surface(min_corner, size, box_count, rng))
    points = np.vstack(points) if points else np.empty((0, 3), dtype=np.float64)

    static_rot = R.from_euler("zyx", [np.pi / 2.0, np.pi / 2.0, 0.0]).as_matrix()
    pose_rot = R.from_euler("zyx", pose[3:]).as_matrix()
    points = points @ static_rot.T + np.array([0.0, 0.0, -gripper_len])
    points = points @ pose_rot.T + pose[:3]
    return points


def create_gripper_cloud_rgb(width_percent, pose, count, rng, gripper_len):
    points = create_gripper_points(width_percent, pose, count, rng, gripper_len=gripper_len)
    colors = np.tile(np.array([[204.0, 51.0, 51.0]], dtype=np.float32), (points.shape[0], 1))
    return np.hstack((points.astype(np.float32), colors))


def create_gripper_cloud_rgb_batch(widths, poses, count, rng, gripper_len):
    gripper_clouds = np.empty((len(poses), count, 6), dtype=np.float32)
    for idx, (width, pose) in enumerate(zip(widths, poses)):
        gripper_clouds[idx] = create_gripper_cloud_rgb(width, pose, count, rng, gripper_len)
    return gripper_clouds


def merge_cloud_with_gripper(original_cloud, gripper_cloud, rng, drop_strategy="random", shuffle_points=True):
    total_points = original_cloud.shape[0]
    gripper_points = min(gripper_cloud.shape[0], total_points)
    keep_points = total_points - gripper_points

    if gripper_points == 0:
        merged = original_cloud.copy()
    elif keep_points == 0:
        idx = rng.choice(gripper_cloud.shape[0], total_points, replace=gripper_cloud.shape[0] < total_points)
        merged = gripper_cloud[idx]
    elif drop_strategy == "tail":
        merged = np.vstack((original_cloud[:keep_points], gripper_cloud[:gripper_points]))
    elif drop_strategy == "near_gripper":
        center = gripper_cloud[:gripper_points, :3].mean(axis=0)
        dist = np.linalg.norm(original_cloud[:, :3] - center, axis=1)
        keep_idx = np.argpartition(dist, gripper_points)[gripper_points:]
        merged = np.vstack((original_cloud[keep_idx], gripper_cloud[:gripper_points]))
    else:
        keep_idx = rng.choice(total_points, keep_points, replace=False)
        merged = np.vstack((original_cloud[keep_idx], gripper_cloud[:gripper_points]))

    if shuffle_points and merged.shape[0] > 1:
        rng.shuffle(merged, axis=0)
    return merged.astype(original_cloud.dtype, copy=False)


def merge_cloud_block_with_gripper(original_block, gripper_block, rng, drop_strategy="tail", shuffle_points=False):
    total_points = original_block.shape[1]
    gripper_points = min(gripper_block.shape[1], total_points)
    keep_points = total_points - gripper_points

    if gripper_points == 0:
        return original_block

    if drop_strategy == "tail" and not shuffle_points:
        original_block[:, keep_points:, :] = gripper_block[:, :gripper_points, :]
        return original_block

    merged = np.empty_like(original_block)
    for frame_idx in range(original_block.shape[0]):
        merged[frame_idx] = merge_cloud_with_gripper(
            original_block[frame_idx],
            gripper_block[frame_idx, :gripper_points],
            rng,
            drop_strategy=drop_strategy,
            shuffle_points=shuffle_points,
        )
    return merged


def selected_camera_names(cloud_group, camera_arg):
    available = list(cloud_group.keys())
    if camera_arg == "all":
        return available
    requested = [name.strip() for name in camera_arg.split(",") if name.strip()]
    missing = [name for name in requested if name not in available]
    if missing:
        raise KeyError(f"Missing camera cloud(s): {missing}; available: {available}")
    return requested


def dataset_addr(dataset):
    return h5py.h5o.get_info(dataset.id).addr


def camera_aliases_by_addr(cloud_group):
    aliases = {}
    for camera_name in cloud_group.keys():
        addr = dataset_addr(cloud_group[camera_name])
        aliases.setdefault(addr, []).append(camera_name)
    return aliases


def dataset_create_kwargs(template, args=None):
    output_compression = "preserve" if args is None else args.output_compression
    kwargs = {}

    if output_compression == "none":
        return kwargs

    if output_compression == "gzip":
        kwargs["compression"] = "gzip"
        kwargs["compression_opts"] = args.gzip_level
        if template.chunks is not None:
            kwargs["chunks"] = template.chunks
        return kwargs

    if template.chunks is not None:
        kwargs["chunks"] = template.chunks
    if template.compression is not None:
        kwargs["compression"] = template.compression
        kwargs["compression_opts"] = template.compression_opts
    if template.shuffle:
        kwargs["shuffle"] = template.shuffle
    if template.fletcher32:
        kwargs["fletcher32"] = template.fletcher32
    if template.scaleoffset is not None:
        kwargs["scaleoffset"] = template.scaleoffset
    if template.fillvalue is not None:
        kwargs["fillvalue"] = template.fillvalue
    if template.maxshape != template.shape:
        kwargs["maxshape"] = template.maxshape
    return kwargs


def create_like_dataset(group, name, template, args=None):
    kwargs = dataset_create_kwargs(template, args)
    new_dataset = group.create_dataset(name, shape=template.shape, dtype=template.dtype, **kwargs)
    for attr_name, attr_value in template.attrs.items():
        new_dataset.attrs[attr_name] = attr_value
    return new_dataset


def choose_batch_frames(dataset, batch_frames):
    if batch_frames and batch_frames > 0:
        return batch_frames
    if dataset.chunks is not None and dataset.chunks[0] > 0:
        return max(1, dataset.chunks[0] * 4)
    return 16


def process_cloud_dataset(read_ds, write_ds, poses, widths, args, rng):
    if read_ds.ndim != 3 or read_ds.shape[-1] != 6:
        raise ValueError(f"cloud dataset must have shape (T, N, 6), got {read_ds.shape}")
    if read_ds.shape[0] != poses.shape[0]:
        raise ValueError(f"cloud frame count {read_ds.shape[0]} != pose frame count {poses.shape[0]}")

    original_points_per_frame = read_ds.shape[1]
    gripper_points = min(args.gripper_points, original_points_per_frame)
    batch_frames = choose_batch_frames(read_ds, args.batch_frames)
    for start in range(0, read_ds.shape[0], batch_frames):
        end = min(start + batch_frames, read_ds.shape[0])
        original_block = read_ds[start:end]
        gripper_block = create_gripper_cloud_rgb_batch(
            widths[start:end],
            poses[start:end],
            gripper_points,
            rng,
            gripper_len=args.gripper_len,
        )
        write_ds[start:end] = merge_cloud_block_with_gripper(
            original_block,
            gripper_block,
            rng,
            drop_strategy=args.drop_strategy,
            shuffle_points=args.shuffle and not args.no_shuffle,
        )
    return read_ds.shape[0], original_points_per_frame, gripper_points


def process_camera_dataset(cloud_group, camera_name, aliases, selected_names, poses, widths, args, rng):
    cloud_ds = cloud_group[camera_name]
    unselected_aliases = [name for name in aliases if name not in selected_names]

    if unselected_aliases:
        tmp_name = f"__tmp_with_gripper_{camera_name}"
        suffix = 0
        while tmp_name in cloud_group:
            suffix += 1
            tmp_name = f"__tmp_with_gripper_{camera_name}_{suffix}"
        new_ds = create_like_dataset(cloud_group, tmp_name, cloud_ds)
        stats = process_cloud_dataset(cloud_ds, new_ds, poses, widths, args, rng)
        del cloud_group[camera_name]
        cloud_group.move(tmp_name, camera_name)
        return stats, f"{camera_name} (separated from aliases: {','.join(unselected_aliases)})"

    stats = process_cloud_dataset(cloud_ds, cloud_ds, poses, widths, args, rng)
    selected_aliases = [name for name in aliases if name in selected_names]
    return stats, ",".join(selected_aliases)


def copy_attrs(src_obj, dst_obj):
    for attr_name, attr_value in src_obj.attrs.items():
        dst_obj.attrs[attr_name] = attr_value


def copy_group_except(src_group, dst_group, current_path, exclude_path):
    copy_attrs(src_group, dst_group)
    for name, obj in src_group.items():
        child_path = f"{current_path}/{name}" if current_path else name
        if child_path == exclude_path:
            continue
        if isinstance(obj, h5py.Group):
            child_group = dst_group.create_group(name)
            copy_group_except(obj, child_group, child_path, exclude_path)
        else:
            src_group.copy(name, dst_group, name=name)


def ensure_group_path(dst_file, path, src_file=None):
    group = dst_file
    current_path = ""
    for part in [part for part in path.split("/") if part]:
        current_path = f"{current_path}/{part}" if current_path else part
        if part not in group:
            group = group.create_group(part)
        else:
            group = group[part]
        if src_file is not None and current_path in src_file:
            copy_attrs(src_file[current_path], group)
    return group


def copy_cloud_aliases(src_cloud_group, dst_cloud_group, aliases):
    first_alias = aliases[0]
    src_cloud_group.copy(first_alias, dst_cloud_group, name=first_alias)
    for alias in aliases[1:]:
        dst_cloud_group[alias] = dst_cloud_group[first_alias]


def add_gripper_to_new_file(src_path, dst_path, args, rng):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        if not args.overwrite:
            print(f"[SKIP] {dst_path} already exists. Use --overwrite to replace it.")
            return False
        dst_path.unlink()

    logs = []
    try:
        with h5py.File(src_path, "r") as src_file, h5py.File(dst_path, "w") as dst_file:
            if args.pose_path not in src_file:
                raise KeyError(f"{args.pose_path} not found in {src_path}")
            if args.eff_angular_path not in src_file:
                raise KeyError(f"{args.eff_angular_path} not found in {src_path}")
            if args.cloud_group_path not in src_file:
                raise KeyError(f"{args.cloud_group_path} not found in {src_path}")

            copy_group_except(src_file, dst_file, "", args.cloud_group_path)

            poses = src_file[args.pose_path][:].astype(np.float32)
            widths = normalize_widths(
                src_file[args.eff_angular_path][:],
                already_normalized=args.eff_angular_is_normalized,
            )
            if poses.shape[0] != widths.shape[0]:
                raise ValueError(f"pose frame count {poses.shape[0]} != eff_angular frame count {widths.shape[0]}")

            src_cloud_group = src_file[args.cloud_group_path]
            dst_cloud_group = ensure_group_path(dst_file, args.cloud_group_path, src_file)
            camera_names = selected_camera_names(src_cloud_group, args.camera)
            selected_names = set(camera_names)
            alias_map = camera_aliases_by_addr(src_cloud_group)

            for aliases in alias_map.values():
                selected_aliases = [name for name in aliases if name in selected_names]
                unselected_aliases = [name for name in aliases if name not in selected_names]

                if unselected_aliases:
                    copy_cloud_aliases(src_cloud_group, dst_cloud_group, unselected_aliases)

                if not selected_aliases:
                    continue

                source_name = selected_aliases[0]
                src_ds = src_cloud_group[source_name]
                dst_ds = create_like_dataset(dst_cloud_group, source_name, src_ds, args)
                stats = process_cloud_dataset(src_ds, dst_ds, poses, widths, args, rng)
                for alias in selected_aliases[1:]:
                    dst_cloud_group[alias] = dst_ds

                frames, original_points_per_frame, gripper_points = stats
                label = ",".join(selected_aliases)
                if unselected_aliases:
                    label += f" (separated from aliases: {','.join(unselected_aliases)})"
                logs.append(
                    f"[OK] {dst_path.name}: {label}, frames={frames}, "
                    f"points/frame={original_points_per_frame}, gripper_points/frame={gripper_points}"
                )
    except Exception:
        if dst_path.exists():
            dst_path.unlink()
        raise

    for line in logs:
        print(line)
    return True


def add_gripper_to_file(src_path, dst_path, args, rng):
    if not args.in_place:
        return add_gripper_to_new_file(src_path, dst_path, args, rng)

    dst_path = src_path

    with h5py.File(dst_path, "r+") as h5_file:
        if args.pose_path not in h5_file:
            raise KeyError(f"{args.pose_path} not found in {dst_path}")
        if args.eff_angular_path not in h5_file:
            raise KeyError(f"{args.eff_angular_path} not found in {dst_path}")
        if args.cloud_group_path not in h5_file:
            raise KeyError(f"{args.cloud_group_path} not found in {dst_path}")

        poses = h5_file[args.pose_path][:].astype(np.float32)
        widths = normalize_widths(
            h5_file[args.eff_angular_path][:],
            already_normalized=args.eff_angular_is_normalized,
        )
        cloud_group = h5_file[args.cloud_group_path]
        camera_names = selected_camera_names(cloud_group, args.camera)
        alias_map = camera_aliases_by_addr(cloud_group)
        selected_names = set(camera_names)
        processed_addrs = set()

        if poses.shape[0] != widths.shape[0]:
            raise ValueError(f"pose frame count {poses.shape[0]} != eff_angular frame count {widths.shape[0]}")

        for camera_name in camera_names:
            cloud_ds = cloud_group[camera_name]
            addr = dataset_addr(cloud_ds)
            aliases = alias_map[addr]
            unselected_aliases = [name for name in aliases if name not in selected_names]
            if addr in processed_addrs and not unselected_aliases:
                continue
            stats, label = process_camera_dataset(
                cloud_group, camera_name, aliases, selected_names, poses, widths, args, rng
            )
            frames, original_points_per_frame, gripper_points = stats
            print(
                f"[OK] {dst_path.name}: {label}, frames={frames}, "
                f"points/frame={original_points_per_frame}, gripper_points/frame={gripper_points}"
            )
            if not unselected_aliases:
                processed_addrs.add(addr)
    return True


def iter_hdf5_files(input_dir, pattern, max_files=None):
    files = sorted(Path(input_dir).glob(pattern), key=natural_key)
    if max_files is not None:
        files = files[:max_files]
    return files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add gripper point clouds to HDF5 cloud_rgb datasets while keeping each frame's point count unchanged."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing source .hdf5 files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for modified copies.")
    parser.add_argument("--pattern", default="*.hdf5", help="Input file glob pattern.")
    parser.add_argument("--camera", default="all", help='Camera name, comma-separated names, or "all".')
    parser.add_argument("--cloud-group-path", default="observations/cloud_rgb")
    parser.add_argument("--pose-path", default="observations/pose_eular")
    parser.add_argument("--eff-angular-path", default="observations/eff_angular")
    parser.add_argument("--gripper-points", type=int, default=500, help="Gripper points added per frame.")
    parser.add_argument("--gripper-len", type=float, default=0.06, help="Offset used by Stage2Editing.update_gripper.")
    parser.add_argument("--eff-angular-is-normalized", action="store_true", help="Use eff_angular directly as width_percent.")
    parser.add_argument(
        "--drop-strategy",
        choices=["random", "tail", "near_gripper"],
        default="tail",
        help="Which original points to remove before adding gripper points.",
    )
    parser.add_argument("--batch-frames", type=int, default=0, help="Frames processed per HDF5 read/write batch; 0 chooses from chunk size.")
    parser.add_argument(
        "--output-compression",
        choices=["preserve", "none", "gzip"],
        default="preserve",
        help='Compression for rewritten cloud datasets. "none" is fastest but creates larger files.',
    )
    parser.add_argument("--gzip-level", type=int, default=4, help="Gzip level when --output-compression gzip is used.")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of HDF5 files to process in parallel.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for point sampling and original-point dropping.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle merged points after appending gripper points.")
    parser.add_argument("--no-shuffle", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--in-place", action="store_true", help="Modify source files directly instead of writing copies.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned work.")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most this many files.")
    return parser.parse_args()


def process_one_file(task):
    idx, src_path, dst_path, args = task
    rng = np.random.default_rng(args.seed + idx)
    return add_gripper_to_file(Path(src_path), Path(dst_path), args, rng)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = iter_hdf5_files(input_dir, args.pattern, args.max_files)

    if not files:
        raise FileNotFoundError(f"No files matched {input_dir / args.pattern}")

    print(f"Input: {input_dir}")
    print("Output: in-place" if args.in_place else f"Output: {output_dir}")
    print(f"Files: {len(files)}")
    print(f"Camera: {args.camera}")
    print(f"Gripper points/frame: {args.gripper_points}")
    print(f"Drop strategy: {args.drop_strategy}, shuffle: {args.shuffle and not args.no_shuffle}")
    print(f"Output compression: {args.output_compression}")
    print(f"Workers: {args.num_workers}")

    if args.dry_run:
        for src_path in files:
            dst_path = src_path if args.in_place else output_dir / src_path.name
            with h5py.File(src_path, "r") as h5_file:
                cameras = selected_camera_names(h5_file[args.cloud_group_path], args.camera)
                shapes = {camera: h5_file[f"{args.cloud_group_path}/{camera}"].shape for camera in cameras}
                aliases = [
                    names for names in camera_aliases_by_addr(h5_file[args.cloud_group_path]).values() if len(names) > 1
                ]
            print(f"[DRY] {src_path} -> {dst_path}: {shapes}, linked_aliases={aliases}")
        return

    tasks = [
        (idx, str(src_path), str(src_path if args.in_place else output_dir / src_path.name), args)
        for idx, src_path in enumerate(files)
    ]

    processed = 0
    if args.num_workers <= 1:
        for task in tasks:
            processed += int(process_one_file(task))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_one_file, task) for task in tasks]
            for future in as_completed(futures):
                processed += int(future.result())

    print(f"Done. Processed {processed}/{len(files)} files.")


if __name__ == "__main__":
    main()

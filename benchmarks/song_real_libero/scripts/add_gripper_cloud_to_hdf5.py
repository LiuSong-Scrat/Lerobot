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
    from .virtual_gripper_geometry import (
        CanonicalGripperParameters,
        GRIPPER_COLOR_RGB,
        GRIPPER_CONTRACT,
        allocate_surface_counts,
        physical_opening_widths,
        sample_gripper_width,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import REAL_DATA_ROOT
    from virtual_gripper_geometry import (
        CanonicalGripperParameters,
        GRIPPER_COLOR_RGB,
        GRIPPER_CONTRACT,
        allocate_surface_counts,
        physical_opening_widths,
        sample_gripper_width,
    )


DEFAULT_INPUT_DIR = "/opt/data/private/liusong/benchmarks/song_real_libero/data/real_setting/hdf5_raw/fold/temp_num2"
DEFAULT_OUTPUT_DIR = "/opt/data/private/liusong/benchmarks/song_real_libero/data/real_setting/hdf5_raw/fold/hdf5_with_gripper"
RH20T_GRIPPER_CONTRACT = GRIPPER_CONTRACT
RH20T_TOTAL_POINTS = 50_000
RH20T_GRIPPER_POINTS = 500
# StaticFranka recordings and LIBERO use the Franka 0.08 m physical opening.
# Other RH20T hardware profiles may override this explicitly (for example
# Robotiq 2F-85 with ``--gripper-max-width-m 0.085``).
DEFAULT_GRIPPER_MAX_WIDTH_M = 0.08
RH20T_GRIPPER_COLOR_RGB = GRIPPER_COLOR_RGB


def natural_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def achieved_gripper_widths(eff_angular, already_normalized=False, max_width_m=DEFAULT_GRIPPER_MAX_WIDTH_M):
    """Return achieved physical two-finger clear-gap widths in metres.

    This follows the RH20T v3 observation rule: virtual geometry is driven by
    the achieved opening, clipped to a fixed physical device limit, rather than
    by per-episode min/max normalization.  Human-hand HDF5 ``eff_angular`` is
    already an achieved parallel-jaw opening in metres.
    """

    return physical_opening_widths(
        eff_angular,
        already_normalized=already_normalized,
        max_width_m=max_width_m,
    )


def reap_gripper_template(opening_width_m, count, rng, gripper_len=0.06, max_width_m=DEFAULT_GRIPPER_MAX_WIDTH_M):
    """Compatibility entry point for the canonical RH20T v3 local template."""

    if not np.isclose(float(gripper_len), 0.06):
        raise ValueError("RH20T v3 fixes the EEF origin; gripper_len must remain 0.06")
    parameters = CanonicalGripperParameters(max_width_m=float(max_width_m))
    return sample_gripper_width(opening_width_m, count, rng, parameters=parameters)


def transform_eef_template_to_reference(points_eef, eef_pose):
    """Apply T_reference<-EEF using [x,y,z,euler_zyx] achieved EEF pose."""

    points_eef = np.asarray(points_eef, dtype=np.float64)
    pose = np.asarray(eef_pose, dtype=np.float64)
    if points_eef.ndim != 2 or points_eef.shape[1] != 3:
        raise ValueError(f"EEF template must have shape (N, 3), got {points_eef.shape}")
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"EEF pose must be one finite [x,y,z,euler_zyx] vector, got {pose}")
    pose_rot = R.from_euler("zyx", pose[3:]).as_matrix()
    # Safe row-vector form: p_reference = p_eef @ R.T + t.
    return points_eef @ pose_rot.T + pose[:3]


def create_gripper_points(opening_width_m, eef_pose, count, rng, gripper_len=0.06, max_width_m=DEFAULT_GRIPPER_MAX_WIDTH_M):
    template_eef = reap_gripper_template(
        opening_width_m, count, rng, gripper_len=gripper_len, max_width_m=max_width_m
    )
    return transform_eef_template_to_reference(template_eef, eef_pose)


def create_gripper_cloud_rgb(opening_width_m, pose, count, rng, gripper_len, max_width_m=DEFAULT_GRIPPER_MAX_WIDTH_M):
    points = create_gripper_points(
        opening_width_m, pose, count, rng, gripper_len=gripper_len, max_width_m=max_width_m
    )
    colors = np.tile(RH20T_GRIPPER_COLOR_RGB[None, :], (points.shape[0], 1))
    return np.hstack((points.astype(np.float32), colors))


def create_gripper_cloud_rgb_batch(widths, poses, count, rng, gripper_len, max_width_m=DEFAULT_GRIPPER_MAX_WIDTH_M):
    gripper_clouds = np.empty((len(poses), count, 6), dtype=np.float32)
    for idx, (width, pose) in enumerate(zip(widths, poses)):
        gripper_clouds[idx] = create_gripper_cloud_rgb(
            width, pose, count, rng, gripper_len, max_width_m=max_width_m
        )
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


def write_cloud_contract_attrs(dataset, args, widths_m):
    dataset.attrs["virtual_gripper_contract"] = RH20T_GRIPPER_CONTRACT
    dataset.attrs["virtual_gripper_template"] = "rh20t_canonical_four_box_v3"
    dataset.attrs["virtual_gripper_pose_source"] = args.pose_path
    dataset.attrs["virtual_gripper_pose_representation"] = "xyz_euler_zyx"
    dataset.attrs["virtual_gripper_width_source"] = args.eff_angular_path
    dataset.attrs["virtual_gripper_width_source_unit"] = (
        "normalized_fraction" if args.eff_angular_is_normalized else "meter"
    )
    dataset.attrs["virtual_gripper_width_geometry_unit"] = "meter"
    dataset.attrs["virtual_gripper_width_geometry_semantics"] = "clear_gap_between_inner_finger_faces"
    dataset.attrs["virtual_gripper_palm_width_m"] = 0.10
    dataset.attrs["virtual_gripper_width_normalized_for_geometry"] = False
    dataset.attrs["virtual_gripper_width_max_m"] = float(args.gripper_max_width_m)
    dataset.attrs["virtual_gripper_width_min_m"] = float(np.min(widths_m))
    dataset.attrs["virtual_gripper_width_observed_max_m"] = float(np.max(widths_m))
    dataset.attrs["virtual_gripper_points"] = int(args.gripper_points)
    dataset.attrs["virtual_gripper_color_rgb"] = RH20T_GRIPPER_COLOR_RGB.astype(np.uint8)
    dataset.attrs["point_cloud_layout"] = (
        f"scene={dataset.shape[1] - int(args.gripper_points)}, "
        f"virtual_gripper_tail={int(args.gripper_points)}"
    )


def write_file_contract_attrs(h5_file, args, widths_m):
    h5_file.attrs["virtual_gripper_contract"] = RH20T_GRIPPER_CONTRACT
    h5_file.attrs["virtual_gripper_template"] = "rh20t_canonical_four_box_v3"
    h5_file.attrs["virtual_gripper_pose_source"] = args.pose_path
    h5_file.attrs["virtual_gripper_pose_semantics"] = "achieved_eef_pose"
    h5_file.attrs["virtual_gripper_width_source"] = args.eff_angular_path
    h5_file.attrs["virtual_gripper_width_semantics"] = "achieved_opening"
    h5_file.attrs["virtual_gripper_width_geometry_unit"] = "meter"
    h5_file.attrs["virtual_gripper_width_geometry_semantics"] = "clear_gap_between_inner_finger_faces"
    h5_file.attrs["virtual_gripper_palm_width_m"] = 0.10
    h5_file.attrs["virtual_gripper_width_normalized_for_geometry"] = False
    h5_file.attrs["virtual_gripper_width_max_m"] = float(args.gripper_max_width_m)
    h5_file.attrs["virtual_gripper_width_min_m"] = float(np.min(widths_m))
    h5_file.attrs["virtual_gripper_width_observed_max_m"] = float(np.max(widths_m))
    h5_file.attrs["virtual_gripper_points"] = int(args.gripper_points)
    h5_file.attrs["point_cloud_total_points"] = RH20T_TOTAL_POINTS
    h5_file.attrs["point_cloud_scene_points"] = RH20T_TOTAL_POINTS - int(args.gripper_points)
    h5_file.attrs["point_cloud_layout"] = (
        f"scene={RH20T_TOTAL_POINTS - int(args.gripper_points)}, "
        f"virtual_gripper_tail={int(args.gripper_points)}"
    )


def process_cloud_dataset(read_ds, write_ds, poses, widths_m, args, rng):
    if read_ds.ndim != 3 or read_ds.shape[-1] != 6:
        raise ValueError(f"cloud dataset must have shape (T, N, 6), got {read_ds.shape}")
    if read_ds.shape[0] != poses.shape[0]:
        raise ValueError(f"cloud frame count {read_ds.shape[0]} != pose frame count {poses.shape[0]}")

    original_points_per_frame = read_ds.shape[1]
    if original_points_per_frame != RH20T_TOTAL_POINTS:
        raise ValueError(
            f"RH20T contract requires {RH20T_TOTAL_POINTS} total points per frame, "
            f"got {original_points_per_frame} in {read_ds.name}"
        )
    gripper_points = int(args.gripper_points)
    batch_frames = choose_batch_frames(read_ds, args.batch_frames)
    for start in range(0, read_ds.shape[0], batch_frames):
        end = min(start + batch_frames, read_ds.shape[0])
        original_block = read_ds[start:end]
        gripper_block = create_gripper_cloud_rgb_batch(
            widths_m[start:end],
            poses[start:end],
            gripper_points,
            rng,
            gripper_len=args.gripper_len,
            max_width_m=args.gripper_max_width_m,
        )
        write_ds[start:end] = merge_cloud_block_with_gripper(
            original_block,
            gripper_block,
            rng,
            drop_strategy=args.drop_strategy,
            shuffle_points=args.shuffle and not args.no_shuffle,
        )
    write_cloud_contract_attrs(write_ds, args, widths_m)
    return read_ds.shape[0], original_points_per_frame, gripper_points


def process_camera_dataset(
    cloud_group,
    camera_name,
    aliases,
    selected_names,
    poses,
    widths_m,
    args,
    rng,
):
    cloud_ds = cloud_group[camera_name]
    unselected_aliases = [name for name in aliases if name not in selected_names]

    if unselected_aliases:
        tmp_name = f"__tmp_with_gripper_{camera_name}"
        suffix = 0
        while tmp_name in cloud_group:
            suffix += 1
            tmp_name = f"__tmp_with_gripper_{camera_name}_{suffix}"
        new_ds = create_like_dataset(cloud_group, tmp_name, cloud_ds)
        stats = process_cloud_dataset(
            cloud_ds, new_ds, poses, widths_m, args, rng
        )
        del cloud_group[camera_name]
        cloud_group.move(tmp_name, camera_name)
        return stats, f"{camera_name} (separated from aliases: {','.join(unselected_aliases)})"

    stats = process_cloud_dataset(
        cloud_ds, cloud_ds, poses, widths_m, args, rng
    )
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
            widths_m = achieved_gripper_widths(
                src_file[args.eff_angular_path][:],
                already_normalized=args.eff_angular_is_normalized,
                max_width_m=args.gripper_max_width_m,
            )
            if poses.ndim != 2 or poses.shape[1] != 6:
                raise ValueError(f"achieved EEF poses must have shape (T, 6), got {poses.shape}")
            if poses.shape[0] != widths_m.shape[0]:
                raise ValueError(
                    f"pose frame count {poses.shape[0]} != eff_angular frame count {widths_m.shape[0]}"
                )

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
                stats = process_cloud_dataset(
                    src_ds,
                    dst_ds,
                    poses,
                    widths_m,
                    args,
                    rng,
                )
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
            write_file_contract_attrs(dst_file, args, widths_m)
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
        widths_m = achieved_gripper_widths(
            h5_file[args.eff_angular_path][:],
            already_normalized=args.eff_angular_is_normalized,
            max_width_m=args.gripper_max_width_m,
        )
        cloud_group = h5_file[args.cloud_group_path]
        camera_names = selected_camera_names(cloud_group, args.camera)
        alias_map = camera_aliases_by_addr(cloud_group)
        selected_names = set(camera_names)
        processed_addrs = set()

        if poses.ndim != 2 or poses.shape[1] != 6:
            raise ValueError(f"achieved EEF poses must have shape (T, 6), got {poses.shape}")
        if poses.shape[0] != widths_m.shape[0]:
            raise ValueError(
                f"pose frame count {poses.shape[0]} != eff_angular frame count {widths_m.shape[0]}"
            )

        for camera_name in camera_names:
            cloud_ds = cloud_group[camera_name]
            addr = dataset_addr(cloud_ds)
            aliases = alias_map[addr]
            unselected_aliases = [name for name in aliases if name not in selected_names]
            if addr in processed_addrs and not unselected_aliases:
                continue
            stats, label = process_camera_dataset(
                cloud_group,
                camera_name,
                aliases,
                selected_names,
                poses,
                widths_m,
                args,
                rng,
            )
            frames, original_points_per_frame, gripper_points = stats
            print(
                f"[OK] {dst_path.name}: {label}, frames={frames}, "
                f"points/frame={original_points_per_frame}, gripper_points/frame={gripper_points}"
            )
            if not unselected_aliases:
                processed_addrs.add(addr)
        write_file_contract_attrs(h5_file, args, widths_m)
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
    parser.add_argument(
        "--gripper-points",
        type=int,
        default=RH20T_GRIPPER_POINTS,
        help=(
            "Virtual-gripper points reserved at the tail of the fixed 50,000-point budget. "
            f"RH20T v3 requires {RH20T_GRIPPER_POINTS}."
        ),
    )
    parser.add_argument("--gripper-len", type=float, default=0.06, help="Offset used by Stage2Editing.update_gripper.")
    parser.add_argument(
        "--gripper-max-width-m",
        type=float,
        default=DEFAULT_GRIPPER_MAX_WIDTH_M,
        help=(
            "Physical achieved-opening limit in metres. Metre-valued eff_angular is passed directly to "
            "geometry and only clipped at this device limit; it is never normalized. "
            f"Default: {DEFAULT_GRIPPER_MAX_WIDTH_M} m for StaticFranka/LIBERO; "
            "override it for another physical gripper profile."
        ),
    )
    parser.add_argument(
        "--eff-angular-is-normalized",
        action="store_true",
        help="Interpret eff_angular as [0,1] achieved-opening fractions instead of metres.",
    )
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
    if args.gripper_points != RH20T_GRIPPER_POINTS:
        raise ValueError(
            f"RH20T v3 requires exactly {RH20T_GRIPPER_POINTS} virtual-gripper points, "
            f"got {args.gripper_points}"
        )
    if args.drop_strategy != "tail" or (args.shuffle and not args.no_shuffle):
        raise ValueError(
            "RH20T v3 requires an ordered point cloud with scene points first and the virtual "
            "gripper in the fixed tail; use --drop-strategy tail without --shuffle."
        )
    if not np.isclose(float(args.gripper_len), 0.06):
        raise ValueError(f"RH20T v3 canonical template requires --gripper-len 0.06, got {args.gripper_len}")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = iter_hdf5_files(input_dir, args.pattern, args.max_files)

    if not files:
        raise FileNotFoundError(f"No files matched {input_dir / args.pattern}")

    print(f"Input: {input_dir}")
    print("Output: in-place" if args.in_place else f"Output: {output_dir}")
    print(f"Files: {len(files)}")
    print(f"Camera: {args.camera}")
    print(f"Gripper contract: {RH20T_GRIPPER_CONTRACT}")
    print(f"Gripper points/frame: {args.gripper_points}")
    print(f"Gripper max achieved width: {args.gripper_max_width_m} m")
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

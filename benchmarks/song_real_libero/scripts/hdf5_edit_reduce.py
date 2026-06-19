#!/usr/bin/env python3
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py


# Edit these values directly, then run:
# python StageGen/scripts/'HDF5_Edit copy.py'
HDF5_DIR = "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/temp/temp_hdf5"
NUM = 2
OUTPUT_DIR = f"/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/temp/temp_num{NUM}"
SUFFIX = f"_num{NUM}"
RECURSIVE = False
OVERWRITE = True
NUM_WORKERS = min(16, os.cpu_count() or 1)
BATCH_FRAMES = 64
KEEP_COMPRESSION = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch downsample hdf5 files by keeping frames with [::NUM]."
    )
    parser.add_argument(
        "hdf5_dir",
        nargs="?",
        default=HDF5_DIR,
        help=f"Folder containing hdf5 files. Default: {HDF5_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=(
            "Folder for generated files. Default: write next to each input "
            f"file with suffix '{SUFFIX}'."
        ),
    )
    parser.add_argument(
        "--suffix",
        default=SUFFIX,
        help=f"Output filename suffix when --output-dir is not set. Default: {SUFFIX}",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=NUM,
        help=(
            "Frame downsample interval. New frame count is about "
            f"All_Frame / NUM. Default: {NUM}."
        ),
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=RECURSIVE,
        help=f"Scan subdirectories recursively. Default: {RECURSIVE}.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=OVERWRITE,
        help=f"Overwrite existing output files. Default: {OVERWRITE}.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of hdf5 files to process in parallel. Default: {NUM_WORKERS}.",
    )
    parser.add_argument(
        "--batch-frames",
        type=int,
        default=BATCH_FRAMES,
        help=f"Frames written per dataset batch. Default: {BATCH_FRAMES}.",
    )
    parser.add_argument(
        "--keep-compression",
        action=argparse.BooleanOptionalAction,
        default=KEEP_COMPRESSION,
        help=(
            "Keep source dataset compression filters. Faster when false, "
            f"smaller output when true. Default: {KEEP_COMPRESSION}."
        ),
    )
    return parser.parse_args()


def is_relative_to(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def iter_hdf5_files(root_dir, recursive, output_dir, suffix):
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"HDF5 folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    output_root = Path(output_dir).expanduser().resolve() if output_dir else None
    candidates = root.rglob("*") if recursive else root.iterdir()

    for path in sorted(candidates):
        if not path.is_file() or path.suffix.lower() not in {".hdf5", ".h5"}:
            continue
        if output_root and is_relative_to(path.resolve(), output_root):
            continue
        if not output_root and suffix and path.stem.endswith(suffix):
            continue
        yield path


def output_path_for(input_path, root_dir, output_dir, suffix):
    if output_dir:
        relative_path = input_path.resolve().relative_to(Path(root_dir).expanduser().resolve())
        return Path(output_dir).expanduser() / relative_path
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def group_for_path(output_file, group_path):
    return output_file.require_group(group_path) if group_path else output_file


def child_path(parent_path, key):
    return f"{parent_path}/{key}" if parent_path else key


def object_identity(obj):
    info = h5py.h5o.get_info(obj.id)
    return info.fileno, info.addr


def collect_key_paths(group, parent_path=""):
    key_paths = set()
    for key in group.keys():
        path = child_path(parent_path, key)
        key_paths.add(path)

        link = group.get(key, getlink=True)
        if not isinstance(link, h5py.HardLink):
            continue

        obj = group.get(key)
        if isinstance(obj, h5py.Group):
            key_paths.update(collect_key_paths(obj, path))
    return key_paths


def collect_attr_paths(obj, parent_path=""):
    attr_paths = {f"{parent_path}@{key}" for key in obj.attrs.keys()}

    if not isinstance(obj, h5py.Group):
        return attr_paths

    for key in obj.keys():
        path = child_path(parent_path, key)
        link = obj.get(key, getlink=True)
        if not isinstance(link, h5py.HardLink):
            continue

        child_obj = obj.get(key)
        if isinstance(child_obj, (h5py.Group, h5py.Dataset)):
            attr_paths.update(collect_attr_paths(child_obj, path))

    return attr_paths


def adjusted_chunks(source_chunks, target_shape):
    if source_chunks is None or not target_shape:
        return None
    return tuple(max(1, min(chunk, dim)) for chunk, dim in zip(source_chunks, target_shape))


def dataset_create_kwargs(source_dataset, target_shape, keep_compression):
    kwargs = {}

    if keep_compression:
        chunks = adjusted_chunks(source_dataset.chunks, target_shape)
        if chunks is not None:
            kwargs["chunks"] = chunks

        if source_dataset.compression is not None:
            kwargs["compression"] = source_dataset.compression
        if source_dataset.compression_opts is not None:
            kwargs["compression_opts"] = source_dataset.compression_opts
        if source_dataset.shuffle:
            kwargs["shuffle"] = source_dataset.shuffle
        if source_dataset.fletcher32:
            kwargs["fletcher32"] = source_dataset.fletcher32
        if source_dataset.scaleoffset is not None:
            kwargs["scaleoffset"] = source_dataset.scaleoffset

    if source_dataset.maxshape and any(dim is None for dim in source_dataset.maxshape):
        kwargs["maxshape"] = source_dataset.maxshape

    return kwargs


def downsampled_count(count, num):
    return (count + num - 1) // num


def downsampled_shape(source_shape, num):
    if source_shape:
        return (downsampled_count(source_shape[0], num),) + source_shape[1:]
    return source_shape


def copy_dataset_by_num(source_dataset, target_dataset, num, batch_frames):
    source_count = source_dataset.shape[0]
    dst_start = 0
    while dst_start < target_dataset.shape[0]:
        src_start = dst_start * num
        src_stop = min(source_count, src_start + batch_frames * num)
        data = source_dataset[src_start:src_stop:num]
        dst_stop = dst_start + data.shape[0]
        target_dataset[dst_start:dst_stop] = data
        dst_start = dst_stop


def create_dataset(output_file, name, source_dataset, num, batch_frames, keep_compression):
    target_shape = downsampled_shape(source_dataset.shape, num)
    parent_path, _, dataset_name = name.rpartition("/")
    parent = group_for_path(output_file, parent_path)

    target_dataset = parent.create_dataset(
        dataset_name,
        shape=target_shape,
        dtype=source_dataset.dtype,
        **dataset_create_kwargs(source_dataset, target_shape, keep_compression),
    )

    if source_dataset.shape:
        copy_dataset_by_num(source_dataset, target_dataset, num, batch_frames)
    else:
        target_dataset[()] = source_dataset[()]

    copy_attrs(source_dataset, target_dataset)


def copy_link(parent_group, key, link):
    if isinstance(link, h5py.SoftLink):
        parent_group[key] = h5py.SoftLink(link.path)
        return True
    if isinstance(link, h5py.ExternalLink):
        parent_group[key] = h5py.ExternalLink(link.filename, link.path)
        return True
    return False


def copy_hdf5_items(
    source_group,
    output_file,
    parent_path,
    num,
    batch_frames,
    keep_compression,
    copied_objects,
):
    output_group = group_for_path(output_file, parent_path)
    stats = {"datasets": 0, "downsampled_datasets": 0}

    for key in source_group.keys():
        path = child_path(parent_path, key)
        link = source_group.get(key, getlink=True)

        if copy_link(output_group, key, link):
            continue

        obj = source_group.get(key)
        obj_identity = object_identity(obj)
        if obj_identity in copied_objects:
            output_group[key] = output_file[copied_objects[obj_identity]]
            continue

        if isinstance(obj, h5py.Group):
            target_group = output_file.require_group(path)
            copied_objects[obj_identity] = path
            copy_attrs(obj, target_group)
            child_stats = copy_hdf5_items(
                obj,
                output_file,
                path,
                num,
                batch_frames,
                keep_compression,
                copied_objects,
            )
            stats["datasets"] += child_stats["datasets"]
            stats["downsampled_datasets"] += child_stats["downsampled_datasets"]
        elif isinstance(obj, h5py.Dataset):
            create_dataset(
                output_file,
                path,
                obj,
                num,
                batch_frames,
                keep_compression,
            )
            copied_objects[obj_identity] = path
            stats["datasets"] += 1
            stats["downsampled_datasets"] += int(bool(obj.shape))
        else:
            source_group.copy(key, output_group, name=key)
            copied_objects[obj_identity] = path

    return stats


def downsample_file(input_path, output_path, overwrite, num, batch_frames, keep_compression):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "input": str(input_path),
            "output": str(output_path),
            "message": f"Skip existing: {output_path}",
        }
    if input_path.resolve() == output_path.resolve():
        raise ValueError(f"Output path is the same as input path: {input_path}")

    start_time = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        copy_attrs(f_in, f_out)
        source_keys = collect_key_paths(f_in)
        source_attr_keys = collect_attr_paths(f_in)

        stats = copy_hdf5_items(
            f_in,
            f_out,
            "",
            num,
            batch_frames,
            keep_compression,
            {},
        )

        output_keys = collect_key_paths(f_out)
        missing_keys = sorted(source_keys - output_keys)
        extra_keys = sorted(output_keys - source_keys)
        if missing_keys or extra_keys:
            raise ValueError(
                "Output hdf5 keys do not match input. "
                f"Missing: {missing_keys[:20]}, Extra: {extra_keys[:20]}"
            )

        output_attr_keys = collect_attr_paths(f_out)
        missing_attr_keys = sorted(source_attr_keys - output_attr_keys)
        extra_attr_keys = sorted(output_attr_keys - source_attr_keys)
        if missing_attr_keys or extra_attr_keys:
            raise ValueError(
                "Output hdf5 attribute keys do not match input. "
                f"Missing: {missing_attr_keys[:20]}, Extra: {extra_attr_keys[:20]}"
            )

    elapsed = time.perf_counter() - start_time
    return {
        "status": "processed",
        "input": str(input_path),
        "output": str(output_path),
        "datasets": stats["datasets"],
        "downsampled_datasets": stats["downsampled_datasets"],
        "key_count": len(source_keys),
        "attr_key_count": len(source_attr_keys),
        "elapsed": elapsed,
    }


def main():
    args = parse_args()
    root = Path(args.hdf5_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    workers = max(1, int(args.workers))
    num = max(1, int(args.num))
    batch_frames = max(1, int(args.batch_frames))

    files = list(iter_hdf5_files(root, args.recursive, output_dir, args.suffix))
    if not files:
        print(f"No hdf5 files found in: {root}")
        return

    tasks = [(input_path, output_path_for(input_path, root, output_dir, args.suffix)) for input_path in files]
    workers = min(workers, len(tasks))
    start_time = time.perf_counter()
    processed = 0
    skipped = 0
    failed = 0

    print(f"Input folder: {root}")
    print(f"Output folder: {output_dir if output_dir else 'same as input'}")
    print(f"Files: {len(tasks)}, workers: {workers}, num: {num}, batch_frames: {batch_frames}")
    print(f"Keep compression: {args.keep_compression}")

    def handle_result(result):
        if result["status"] == "skipped":
            print(result["message"])
            return "skipped"

        print(
            f"{Path(result['input']).name}: "
            f"datasets={result['datasets']}, "
            f"downsampled={result['downsampled_datasets']}, "
            f"keys={result['key_count']}, "
            f"attr_keys={result['attr_key_count']}, "
            f"{result['elapsed']:.2f}s"
        )
        print(f"  saved: {result['output']}")
        return "processed"

    if workers == 1:
        for input_path, output_path in tasks:
            try:
                status = handle_result(
                    downsample_file(
                        input_path,
                        output_path,
                        args.overwrite,
                        num,
                        batch_frames,
                        args.keep_compression,
                    )
                )
                processed += status == "processed"
                skipped += status == "skipped"
            except Exception as exc:
                failed += 1
                print(f"Failed: {input_path.name}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    downsample_file,
                    input_path,
                    output_path,
                    args.overwrite,
                    num,
                    batch_frames,
                    args.keep_compression,
                ): input_path
                for input_path, output_path in tasks
            }
            for future in as_completed(futures):
                input_path = futures[future]
                try:
                    status = handle_result(future.result())
                    processed += status == "processed"
                    skipped += status == "skipped"
                except Exception as exc:
                    failed += 1
                    print(f"Failed: {input_path.name}: {exc}")

    elapsed = time.perf_counter() - start_time
    print(
        f"Done. Processed {processed}, skipped {skipped}, "
        f"failed {failed}, total {elapsed:.2f}s."
    )


if __name__ == "__main__":
    main()

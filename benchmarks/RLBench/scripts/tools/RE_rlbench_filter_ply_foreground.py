#!/usr/bin/env python3
"""Remove high-foreground-score vertices from an ASCII PLY diagnostic.

The cache preview PLY contains scene vertices with a finite
``foreground_score`` and action-trajectory vertices with NaN. Only finite
scores participate in filtering; trajectory vertices and their valid edges
are retained.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Edit these three values, then run this file without command-line arguments.
INPUT_PLY = Path(
    "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/datasets/"
    "rlbench_water_lerobot_20260803_234156_pointseg_cache/visualizations/"
    "episode_000001/p75_frame_000112_soft.ply"
)
OUTPUT_PLY = Path(
    "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/datasets/"
    "rlbench_water_lerobot_20260803_234156_pointseg_cache/visualizations/"
    "episode_000001/p75_frame_000112_soft_drop_top7500fg.ply"
)
REMOVE_TOP_COUNT = 7500


@dataclass
class Element:
    name: str
    count: int
    header_index: int
    properties: list[tuple[str, str, str | None]] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove the highest-foreground-score points from an ASCII PLY."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Optional input path. If omitted, INPUT_PLY above is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. If omitted, OUTPUT_PLY above is used.",
    )
    parser.add_argument(
        "--remove-fraction",
        type=float,
        default=None,
        help="Remove this fraction of the highest foreground-score points.",
    )
    parser.add_argument(
        "--keep-top-fraction",
        type=float,
        default=None,
        help="Keep this fraction of the highest foreground-score points and remove the lower scores.",
    )
    parser.add_argument(
        "--remove-count",
        type=int,
        default=None,
        help="Remove this exact number of finite-score points instead of REMOVE_TOP_COUNT.",
    )
    return parser.parse_args()


def _parse_header(lines: list[str]) -> tuple[list[str], list[Element], int]:
    elements: list[Element] = []
    current: Element | None = None
    end_header = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        match = re.fullmatch(r"element\s+(\S+)\s+(\d+)", stripped)
        if match:
            current = Element(match.group(1), int(match.group(2)), index)
            elements.append(current)
            continue
        if stripped == "end_header":
            end_header = index
            break
        if current is not None and stripped.startswith("property "):
            parts = stripped.split()
            if len(parts) == 3:
                current.properties.append((parts[2], parts[1], None))
            elif len(parts) == 5 and parts[1] == "list":
                current.properties.append((parts[4], parts[3], parts[2]))
    if end_header is None:
        raise ValueError("PLY header has no end_header line.")
    if not any(element.name == "vertex" for element in elements):
        raise ValueError("PLY has no vertex element.")
    return lines[: end_header + 1], elements, end_header + 1


def _tokens_for_record(tokens: list[str], properties: list[tuple[str, str, str | None]]) -> int:
    position = 0
    for _, _, list_count_type in properties:
        if list_count_type is None:
            position += 1
        else:
            if position >= len(tokens):
                raise ValueError("Malformed PLY list property: missing list length.")
            list_length = int(tokens[position])
            position += 1 + list_length
    return position


def _read_ply(path: Path) -> tuple[list[str], list[Element]]:
    with path.open("r", encoding="ascii") as handle:
        lines = handle.read().splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"Not a PLY file: {path}")
    header, elements, data_start = _parse_header(lines)
    if not any(line.strip() == "format ascii 1.0" for line in header):
        raise ValueError("Only ASCII PLY files are supported.")

    cursor = data_start
    for element in elements:
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        for _ in range(element.count):
            if cursor >= len(lines):
                raise ValueError(f"PLY ended while reading element {element.name}.")
            tokens = lines[cursor].split()
            expected = _tokens_for_record(tokens, element.properties)
            if expected != len(tokens):
                raise ValueError(
                    f"Malformed {element.name} record: expected {expected} values, got {len(tokens)}."
                )
            element.rows.append(tokens)
            cursor += 1
    return header, elements


def _property_index(element: Element, name: str) -> int:
    for index, (property_name, _, list_count_type) in enumerate(element.properties):
        if property_name == name and list_count_type is None:
            return index
    raise ValueError(f"Element {element.name!r} has no scalar property {name!r}.")


def _default_output(path: Path) -> Path:
    return path.with_name(f"{path.stem}_drop_top50fg{path.suffix}")


def filter_ply(
    input_path: Path,
    output_path: Path,
    *,
    remove_count: int | None = None,
    remove_fraction: float | None = None,
    keep_top_fraction: float | None = None,
) -> tuple[int, int, float]:
    selected_options = sum(
        value is not None
        for value in (remove_count, remove_fraction, keep_top_fraction)
    )
    if selected_options > 1:
        raise ValueError(
            "Use only one of remove_count, remove_fraction, and keep_top_fraction."
        )
    if remove_count is not None and remove_count < 0:
        raise ValueError("remove_count must be non-negative.")
    if remove_fraction is not None and not 0.0 <= remove_fraction <= 1.0:
        raise ValueError("remove_fraction must be between 0 and 1.")
    if keep_top_fraction is not None and not 0.0 <= keep_top_fraction <= 1.0:
        raise ValueError("keep_top_fraction must be between 0 and 1.")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from input path.")

    header, elements = _read_ply(input_path)
    vertex = next(element for element in elements if element.name == "vertex")
    score_index = _property_index(vertex, "foreground_score")
    try:
        scores = np.asarray([float(row[score_index]) for row in vertex.rows], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("foreground_score contains a non-numeric value.") from exc

    finite_indices = np.flatnonzero(np.isfinite(scores))
    if keep_top_fraction is not None:
        keep_count = int(np.ceil(finite_indices.size * float(keep_top_fraction)))
        remove_count = finite_indices.size - keep_count
        ascending = finite_indices[np.argsort(scores[finite_indices], kind="stable")]
        remove_indices = ascending[:remove_count]
    elif remove_count is None:
        remove_count = int(np.floor(finite_indices.size * float(remove_fraction)))
        if remove_count > finite_indices.size:
            raise ValueError(
                f"Requested removal of {remove_count} points, but only "
                f"{finite_indices.size} finite-score points exist."
            )
        descending = finite_indices[np.argsort(-scores[finite_indices], kind="stable")]
        remove_indices = descending[:remove_count]
    else:
        if remove_count > finite_indices.size:
            raise ValueError(
                f"Requested removal of {remove_count} points, but only "
                f"{finite_indices.size} finite-score points exist."
            )
        descending = finite_indices[np.argsort(-scores[finite_indices], kind="stable")]
        remove_indices = descending[:remove_count]
    keep = np.ones(len(vertex.rows), dtype=bool)
    keep[remove_indices] = False

    old_to_new = np.full(len(vertex.rows), -1, dtype=np.int64)
    old_to_new[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    vertex.rows = [row for index, row in enumerate(vertex.rows) if keep[index]]
    old_vertex_count = vertex.count
    vertex.count = len(vertex.rows)

    for element in elements:
        if element.name != "edge":
            continue
        try:
            start_index = _property_index(element, "vertex1")
            end_index = _property_index(element, "vertex2")
        except ValueError:
            continue
        kept_edges: list[list[str]] = []
        for row in element.rows:
            start = int(row[start_index])
            end = int(row[end_index])
            if not (0 <= start < old_vertex_count and 0 <= end < old_vertex_count):
                raise ValueError("Edge references a vertex outside the PLY vertex range.")
            if old_to_new[start] < 0 or old_to_new[end] < 0:
                continue
            row = list(row)
            row[start_index] = str(int(old_to_new[start]))
            row[end_index] = str(int(old_to_new[end]))
            kept_edges.append(row)
        element.rows = kept_edges
        element.count = len(kept_edges)

    header_out = list(header)
    for element in elements:
        header_out[element.header_index] = f"element {element.name} {element.count}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as handle:
        for line in header_out:
            handle.write(line + "\n")
        for element in elements:
            for row in element.rows:
                handle.write(" ".join(row) + "\n")

    removed_threshold = float(scores[remove_indices[-1]]) if remove_indices.size else float("nan")
    return len(remove_indices), len(vertex.rows), removed_threshold


def main() -> None:
    args = parse_args()
    input_path = (args.input or INPUT_PLY).expanduser().resolve()
    output_path = (args.output or (OUTPUT_PLY if args.input is None else _default_output(input_path))).expanduser().resolve()
    if args.remove_count is not None:
        remove_count = args.remove_count
        remove_fraction = None
        keep_top_fraction = None
    elif args.remove_fraction is not None:
        remove_count = None
        remove_fraction = args.remove_fraction
        keep_top_fraction = None
    elif args.keep_top_fraction is not None:
        remove_count = None
        remove_fraction = None
        keep_top_fraction = args.keep_top_fraction
    else:
        remove_count = REMOVE_TOP_COUNT
        remove_fraction = None
        keep_top_fraction = None
    removed, kept, threshold = filter_ply(
        input_path,
        output_path,
        remove_count=remove_count,
        remove_fraction=remove_fraction,
        keep_top_fraction=keep_top_fraction,
    )
    threshold_text = "n/a" if not np.isfinite(threshold) else f"{threshold:.6f}"
    print(f"[input]  {input_path}")
    print(f"[output] {output_path}")
    print(f"[filter] removed={removed} kept={kept} removed_score_min={threshold_text}")


if __name__ == "__main__":
    main()

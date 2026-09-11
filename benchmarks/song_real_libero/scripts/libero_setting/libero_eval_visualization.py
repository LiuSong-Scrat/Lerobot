"""Output-only PLY/RGB rendering, ported from RE_rlbench_official_eval.py.

No simulator stepping, policy inference, or model-input mutation occurs here.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .libero_pointcloud_utils import pose9_to_homo_np


def output_font_path(bold=False):
    """Use bundled DejaVu on headless hosts without system fonts."""
    import sys

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    paths = [Path("/usr/share/fonts/truetype/dejavu") / name]
    paths.extend(
        Path(root) / "matplotlib/mpl-data/fonts/ttf" / name for root in sys.path
    )
    return str(next((path for path in paths if path.is_file()), paths[0]))


def libero_front_observation(env, raw_obs, camera_name="agentview"):
    """Calibration for the saved image, inverse of backproject_camera()."""
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    image = np.asarray(raw_obs[f"{camera_name}_image"])
    height, width = image.shape[:2]
    intrinsic = get_camera_intrinsic_matrix(env.sim, camera_name, height, width)
    # Backprojection uses v_projected = H-1-v_raw. Saved agentview also flips u.
    image_axes = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]])
    if camera_name == "agentview":
        image_axes[0] = [-1.0, 0.0, width - 1.0]
        image = image[:, ::-1]
    return SimpleNamespace(
        front_rgb=np.ascontiguousarray(image),
        misc={
            "front_camera_extrinsics": get_camera_extrinsic_matrix(
                env.sim, camera_name
            ),
            "front_camera_intrinsics": image_axes @ intrinsic,
        },
    )


def canonical_gripper_boxes(width, max_width=0.08):
    """Visualization-only wireframe of the existing RH20T geometry (no re-sampling)."""
    from .libero_pointcloud_utils import CanonicalGripperParameters

    p = CanonicalGripperParameters(max_width_m=max_width)
    width = float(np.clip(width, 0, p.max_width_m))
    t, d, z = p.finger_thickness_m, p.palm_depth_m, -p.finger_length_m * 5 / 8
    corners = [
        (-t / 2, -width / 2 - t, z),
        (-t / 2, width / 2, z),
        (-t / 2, -p.palm_width_m / 2, z - d),
        (-d / 2, -d / 2, z - d - p.handle_length_m),
    ]
    sizes = [
        (t, t, p.finger_length_m),
        (t, t, p.finger_length_m),
        (t, p.palm_width_m, d),
        (d, d, p.handle_length_m),
    ]
    cube = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float32,
    )
    return [
        cube * np.asarray(size) + np.asarray(corner)
        for corner, size in zip(corners, sizes, strict=True)
    ]


def annotate_final_task_result(frame, success):
    """Draw the final episode result in the upper-right corner of one frame."""
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    padding = max(5, int(round(image.width / 100.0)))
    font_size = max(12, min(28, int(round(image.width / 22.0))))
    font_path = output_font_path(bold=True)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        font = ImageFont.load_default()

    label = "SUCCESS: TRUE" if success else "SUCCESS: FALSE"
    color = (45, 225, 80) if success else (255, 55, 45)
    text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    panel_width = text_width + 2 * padding
    panel_height = text_height + 2 * padding
    panel_x = max(0, image.width - panel_width - padding)
    panel_y = padding
    draw.rectangle(
        (panel_x, panel_y, panel_x + panel_width, panel_y + panel_height),
        fill=(0, 0, 0),
        outline=color,
        width=max(2, padding // 2),
    )
    draw.text(
        (panel_x + padding, panel_y + padding - text_box[1]),
        label,
        fill=color,
        font=font,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    return np.asarray(image, dtype=np.uint8)


def foreground_score_colors(
    scores: np.ndarray, base_rgb: np.ndarray | None = None
) -> np.ndarray:
    """Blue -> cyan -> yellow -> red foreground-probability heat map."""
    scores = np.clip(np.asarray(scores, dtype=np.float32).reshape(-1), 0.0, 1.0)
    anchors_x = np.asarray([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
    anchors_rgb = np.asarray(
        [
            [0.05, 0.10, 0.95],
            [0.00, 0.90, 0.95],
            [1.00, 0.90, 0.05],
            [0.95, 0.05, 0.02],
        ],
        dtype=np.float32,
    )
    colors = np.stack(
        [np.interp(scores, anchors_x, anchors_rgb[:, channel]) for channel in range(3)],
        axis=-1,
    ).astype(np.float32)
    if base_rgb is not None:
        base = np.asarray(base_rgb, dtype=np.float32)
        if base.max(initial=0.0) > 1.0:
            base = base / 255.0
        colors = 0.12 * np.clip(base, 0.0, 1.0) + 0.88 * colors
    return np.clip(colors, 0.0, 1.0)


def rgb_to_uint8(rgb):
    """Normalize RLBench RGB observations without truncating [0, 1] floats."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected an HxWx3 RGB observation, got " + str(image.shape))
    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        if image.size and float(np.max(image)) <= 1.0 + 1e-6:
            image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def annotate_video_frame(
    frame,
    task_name,
    episode_index,
    frame_index,
    physics_frame_index,
    model_call,
    chunk_row=None,
):
    """Add readable control-frame metadata to the top of a saved video frame."""
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    padding = max(4, int(round(image.width / 128.0)))
    font_size = max(9, min(20, int(round(image.width / 32.0))))
    font_path = output_font_path()
    while True:
        lines = [
            "task="
            + str(task_name)
            + " episode="
            + str(episode_index).zfill(3)
            + " frame="
            + str(frame_index).zfill(6),
            "physics_frame="
            + str(physics_frame_index).zfill(6)
            + " model_call="
            + str(model_call).zfill(4),
        ]
        if chunk_row is not None:
            lines.append("chunk_row=" + str(chunk_row).zfill(2))
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.load_default()
            break
        widths = [draw.textbbox((0, 0), line, font=font)[2] for line in lines]
        if max(widths) + 2 * padding <= image.width or font_size <= 8:
            break
        font_size -= 1

    line_height = max(draw.textbbox((0, 0), line, font=font)[3] for line in lines)
    line_gap = max(2, padding // 2)
    bar_height = 2 * padding + len(lines) * line_height + (len(lines) - 1) * line_gap
    draw.rectangle((0, 0, image.width, bar_height), fill=(0, 0, 0))
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_height + line_gap
    return np.asarray(image, dtype=np.uint8)


def action_time_colors(count):
    """Return LIBERO-style blue/cyan -> green -> red chunk colors."""
    if count <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    if count == 1:
        return np.asarray([[26, 217, 64]], dtype=np.uint8)
    values = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    start = np.asarray([13.0, 140.0, 255.0], dtype=np.float32)
    middle = np.asarray([26.0, 217.0, 64.0], dtype=np.float32)
    end = np.asarray([255.0, 46.0, 13.0], dtype=np.float32)
    first_half = (1.0 - 2.0 * values) * start + (2.0 * values) * middle
    second_half = (2.0 - 2.0 * values) * middle + (2.0 * values - 1.0) * end
    colors = np.where(values <= 0.5, first_half, second_half)
    return np.clip(colors, 0.0, 255.0).astype(np.uint8)


def sample_colored_line(start, end, start_color, end_color, spacing=0.0015):
    """Represent one line segment as dense colored PLY points."""
    distance = float(np.linalg.norm(end - start))
    count = max(2, int(np.ceil(distance / max(float(spacing), 1e-5))) + 1)
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    points = (1.0 - alpha) * start + alpha * end
    colors = (1.0 - alpha) * start_color + alpha * end_color
    return points.astype(np.float32), colors.astype(np.uint8)


def _foreground_scores_for_cloud(point_cloud, snapshot):
    """Return LitePT foreground scores in the same order as the model input cloud."""
    cloud = np.asarray(point_cloud, dtype=np.float32)
    scores = np.full(len(cloud), np.nan, dtype=np.float32)
    if not isinstance(snapshot, dict):
        return scores

    snapshot_cloud = snapshot.get("point_cloud")
    operation_prob = snapshot.get("operation_prob")
    point_is_pad = snapshot.get("point_is_pad")
    if snapshot_cloud is None or operation_prob is None:
        return scores

    if hasattr(snapshot_cloud, "detach"):
        snapshot_cloud = snapshot_cloud.detach().float().cpu().numpy()
    if hasattr(operation_prob, "detach"):
        operation_prob = operation_prob.detach().float().cpu().numpy()
    if hasattr(point_is_pad, "detach"):
        point_is_pad = point_is_pad.detach().bool().cpu().numpy()

    snapshot_cloud = np.asarray(snapshot_cloud, dtype=np.float32)
    operation_prob = np.asarray(operation_prob, dtype=np.float32).reshape(-1)
    if snapshot_cloud.ndim == 3:
        snapshot_cloud = snapshot_cloud[0]
    if point_is_pad is not None:
        point_is_pad = np.asarray(point_is_pad, dtype=bool).reshape(-1)

    # build_model_batch sends this exact point order to LitePT.  Refuse an
    # accidental positional mismatch instead of painting unrelated points.
    if snapshot_cloud.ndim != 2 or len(snapshot_cloud) != len(cloud):
        return scores
    if not np.allclose(snapshot_cloud[:, :3], cloud[:, :3], atol=1e-5, rtol=1e-5):
        return scores
    if len(operation_prob) != len(cloud):
        return scores
    valid = np.isfinite(operation_prob)
    if point_is_pad is not None and len(point_is_pad) == len(cloud):
        valid = valid & ~point_is_pad
    scores[valid] = np.clip(operation_prob[valid], 0.0, 1.0)
    return scores


def build_action_chunk_ply_cloud(
    point_cloud,
    action_chunk,
    max_points,
    foreground_snapshot=None,
    point_mode="prob",
    execution_start=0,
    execution_stop=0,
):
    """Add EEF paths and orientation axes to the current EEF cloud."""
    cloud = np.asarray(point_cloud, dtype=np.float32)
    actions = np.asarray(action_chunk, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError("Point cloud must have shape (N, >=3).")
    if actions.ndim != 2 or actions.shape[1] < 9:
        raise ValueError("Action chunk must have shape (T, >=9).")

    valid_cloud = np.isfinite(cloud[:, :3]).all(axis=1)
    scene_xyz = cloud[valid_cloud, :3]
    scene_source_indices = np.flatnonzero(valid_cloud).astype(np.int32)
    if point_mode not in {"full", "prob"}:
        raise ValueError("point_mode must be 'full' or 'prob'.")
    foreground_scores = _foreground_scores_for_cloud(cloud, foreground_snapshot)
    if cloud.shape[1] >= 6:
        scene_rgb = cloud[valid_cloud, 3:6]
        if scene_rgb.size and float(np.max(scene_rgb)) <= 1.0:
            scene_rgb = scene_rgb * 255.0
        scene_rgb = np.clip(scene_rgb, 0.0, 255.0).astype(np.uint8)
    else:
        scene_rgb = np.full((len(scene_xyz), 3), 128, dtype=np.uint8)
    scene_scores = foreground_scores[valid_cloud]
    score_valid = np.isfinite(scene_scores)
    if point_mode == "prob" and np.any(score_valid):
        colored = foreground_score_colors(
            scene_scores[score_valid], scene_rgb[score_valid]
        )
        scene_rgb[score_valid] = np.rint(colored * 255.0).astype(np.uint8)

    valid_action_indices = np.flatnonzero(
        np.isfinite(actions[:, :9]).all(axis=1)
    ).astype(np.int32)
    poses = pose9_to_homo_np(actions[valid_action_indices, :9])
    positions = poses[:, :3, 3]
    colors = action_time_colors(len(positions))
    overlay_xyz = []
    overlay_rgb = []
    overlay_kind = []
    overlay_action_index = []
    overlay_phase = []

    def append_overlay(points, point_colors, kind, action_index):
        points = np.asarray(points, dtype=np.float32)
        point_colors = np.array(point_colors, dtype=np.uint8, copy=True)
        phase = 1 if execution_start <= action_index < execution_stop else 2
        if kind == 2 and phase == 2:
            point_colors[:] = np.asarray([150, 150, 150], dtype=np.uint8)
        overlay_xyz.append(points)
        overlay_rgb.append(point_colors)
        overlay_kind.append(np.full(len(points), kind, dtype=np.uint8))
        overlay_action_index.append(
            np.full(len(points), int(action_index), dtype=np.int32)
        )
        overlay_phase.append(np.full(len(points), phase, dtype=np.uint8))

    for local_index in range(max(0, len(positions) - 1)):
        action_index = int(valid_action_indices[local_index])
        next_action_index = int(valid_action_indices[local_index + 1])
        points, point_colors = sample_colored_line(
            positions[local_index],
            positions[local_index + 1],
            colors[local_index].astype(np.float32),
            colors[local_index + 1].astype(np.float32),
        )
        append_overlay(points, point_colors, 1, action_index)
        if next_action_index != action_index + 1:
            raise ValueError("Action chunk contains a non-contiguous valid row.")

    # Every action gets a colored cross and orientation triad.
    marker_offsets = np.linspace(-0.004, 0.004, 9, dtype=np.float32)
    axis_offsets = np.linspace(0.0, 0.018, 8, dtype=np.float32)
    axis_colors = np.asarray(
        [[255, 30, 30], [30, 255, 30], [30, 100, 255]], dtype=np.uint8
    )
    for local_index, action_index_value in enumerate(valid_action_indices):
        action_index = int(action_index_value)
        marker_points = []
        for axis in range(3):
            points = np.repeat(
                positions[local_index][None, :], len(marker_offsets), axis=0
            )
            points[:, axis] += marker_offsets
            marker_points.append(points)
        marker_points = np.concatenate(marker_points, axis=0)
        append_overlay(
            marker_points,
            np.repeat(colors[local_index][None, :], len(marker_points), axis=0),
            1,
            action_index,
        )

        for axis in range(3):
            direction = poses[local_index, :3, axis]
            axis_points = (
                positions[local_index][None, :]
                + axis_offsets[:, None] * direction[None, :]
            )
            append_overlay(
                axis_points,
                np.repeat(axis_colors[axis][None, :], len(axis_points), axis=0),
                1,
                action_index,
            )

    if overlay_xyz:
        trajectory_xyz = np.concatenate(overlay_xyz, axis=0).astype(np.float32)
        trajectory_rgb = np.concatenate(overlay_rgb, axis=0).astype(np.uint8)
        trajectory_kind = np.concatenate(overlay_kind, axis=0)
        trajectory_action_index = np.concatenate(overlay_action_index, axis=0)
        trajectory_phase = np.concatenate(overlay_phase, axis=0)
    else:
        trajectory_xyz = np.empty((0, 3), dtype=np.float32)
        trajectory_rgb = np.empty((0, 3), dtype=np.uint8)
        trajectory_kind = np.empty((0,), dtype=np.uint8)
        trajectory_action_index = np.empty((0,), dtype=np.int32)
        trajectory_phase = np.empty((0,), dtype=np.uint8)

    max_points = max(int(max_points), len(trajectory_xyz))
    scene_limit = max_points - len(trajectory_xyz)
    if len(scene_xyz) > scene_limit:
        indices = np.linspace(0, len(scene_xyz) - 1, scene_limit, dtype=np.int64)
        scene_xyz = scene_xyz[indices]
        scene_rgb = scene_rgb[indices]
        scene_scores = scene_scores[indices]
        scene_source_indices = scene_source_indices[indices]
    xyz = np.concatenate((scene_xyz, trajectory_xyz), axis=0)
    rgb = np.concatenate((scene_rgb, trajectory_rgb), axis=0)
    scores = np.concatenate(
        (scene_scores, np.full(len(trajectory_xyz), np.nan, dtype=np.float32)), axis=0
    )
    source_indices = np.concatenate(
        (
            scene_source_indices,
            np.full(len(trajectory_xyz), -1, dtype=np.int32),
        ),
        axis=0,
    )
    point_kind = np.concatenate(
        (
            np.zeros(len(scene_xyz), dtype=np.uint8),
            trajectory_kind,
        ),
        axis=0,
    )
    action_indices = np.concatenate(
        (
            np.full(len(scene_xyz), -1, dtype=np.int32),
            trajectory_action_index,
        ),
        axis=0,
    )
    action_phase = np.concatenate(
        (
            np.zeros(len(scene_xyz), dtype=np.uint8),
            trajectory_phase,
        ),
        axis=0,
    )
    return xyz, rgb, scores, source_indices, point_kind, action_indices, action_phase


def write_colored_ply(
    path,
    xyz,
    rgb,
    foreground_scores=None,
    source_indices=None,
    point_kind=None,
    action_indices=None,
    action_phase=None,
    point_mode="prob",
    header_comments=(),
):
    """Write action RGB plus LitePT score fields in an Open3D/MeshLab PLY."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("comment coordinate_frame current_virtual_tcp_at_model_call\n")
        file.write(
            "comment coordinate_origin rh20t_virtual_gripper_finger_five_eighths\n"
        )
        for comment in header_comments:
            normalized_comment = str(comment).replace("\n", " ").replace("\r", " ")
            file.write("comment " + normalized_comment + "\n")
        file.write("comment action_path blue_green_red means first_to_last_chunk_row\n")
        file.write(
            "comment point_kind 0=scene_or_current_gripper " "1=eef_path_and_axes\n"
        )
        file.write(
            "comment action_phase 0=scene "
            "1=current_execution_window 2=future_forecast\n"
        )
        file.write("comment scene_point_mode " + str(point_mode) + "\n")
        file.write("element vertex " + str(len(xyz)) + "\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if foreground_scores is not None:
            file.write("property float operation_prob\n")
            file.write("property float selection_score\n")
            file.write("property int source_point_index\n")
            file.write("property uchar point_kind\n")
            file.write("property int action_index\n")
            file.write("property uchar action_phase\n")
        file.write("end_header\n")
        for index, (point, color) in enumerate(zip(xyz, rgb, strict=True)):
            line = f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} {int(color[0])} {int(color[1])} {int(color[2])}"
            if foreground_scores is not None:
                score = float(foreground_scores[index])
                source = (
                    int(source_indices[index]) if source_indices is not None else index
                )
                kind = int(point_kind[index]) if point_kind is not None else 0
                action = (
                    int(action_indices[index]) if action_indices is not None else -1
                )
                phase = int(action_phase[index]) if action_phase is not None else 0
                line += f" {score:.7f} {score:.7f} {source} {kind} {action} {phase}"
            file.write(line + "\n")


def write_model_input_ply_binary(path, point_cloud, comments=()):
    """Write one raw XYZRGB model-input cloud as a compact binary PLY."""
    cloud = np.asarray(point_cloud, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < 6:
        raise ValueError("Expected point cloud shape (N, >=6), got " + str(cloud.shape))
    xyz = cloud[:, :3]
    rgb = cloud[:, 3:6]
    if rgb.size and float(np.nanmax(rgb)) <= 1.0 + 1e-6:
        rgb = rgb * 255.0
    rgb = np.clip(
        np.rint(np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)),
        0.0,
        255.0,
    ).astype(np.uint8)
    valid = np.isfinite(xyz).all(axis=1)
    xyz = np.asarray(xyz[valid], dtype="<f4")
    rgb = rgb[valid]
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
        align=False,
    )
    vertices = np.empty(len(xyz), dtype=vertex_dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    header = [
        "ply",
        "format binary_little_endian 1.0",
        *["comment " + str(comment) for comment in comments],
        "element vertex " + str(len(vertices)),
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        file.write("\n".join(header).encode("ascii"))
        file.write(vertices.tobytes(order="C"))


def project_world_points_to_front_image(world_points, observation):
    """Project world XYZ using the supplied saved-image optical calibration."""
    extrinsics = np.asarray(
        observation.misc["front_camera_extrinsics"], dtype=np.float64
    )
    intrinsics = np.asarray(
        observation.misc["front_camera_intrinsics"], dtype=np.float64
    )
    points = np.asarray(world_points, dtype=np.float64)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera_points = (np.linalg.inv(extrinsics) @ homogeneous.T).T[:, :3]
    projected = (intrinsics @ camera_points.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    height, width = np.asarray(observation.front_rgb).shape[:2]
    valid = np.isfinite(pixels).all(axis=1)
    # Positive optical Z is depth. K already incorporates saved-image flips.
    valid = valid & (camera_points[:, 2] > 1e-6)
    valid = valid & (pixels[:, 0] >= 0.0) & (pixels[:, 0] < width)
    valid = valid & (pixels[:, 1] >= 0.0) & (pixels[:, 1] < height)
    return pixels.astype(np.float32), valid


def draw_action_chunk_on_front_image(
    path,
    observation,
    world_targets,
    image_width,
    execution_start,
    execution_stop,
    current_gripper_points_world=None,
    current_gripper_boxes_world=None,
):
    """Draw the target path and current virtual gripper on the front RGB image."""
    source = Image.fromarray(rgb_to_uint8(observation.front_rgb))
    target_width = int(image_width)
    target_height = max(1, int(round(source.height * target_width / source.width)))
    image = source.resize((target_width, target_height), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    colors = action_time_colors(len(world_targets))
    positions = world_targets[:, :3, 3]
    direction_ends = positions + world_targets[:, :3, 0] * 0.025
    position_pixels, position_valid = project_world_points_to_front_image(
        positions, observation
    )
    direction_pixels, direction_valid = project_world_points_to_front_image(
        direction_ends, observation
    )
    scale = np.asarray(
        [target_width / source.width, target_height / source.height], dtype=np.float32
    )
    scaled = position_pixels * scale
    scaled_direction = direction_pixels * scale

    def phase_color(index):
        if execution_start <= index < execution_stop:
            return tuple(int(value) for value in colors[index])
        return (150, 150, 150)

    def draw_dashed_line(start_point, end_point, color, width):
        start_point = np.asarray(start_point, dtype=np.float32)
        end_point = np.asarray(end_point, dtype=np.float32)
        distance = float(np.linalg.norm(end_point - start_point))
        segments = max(1, int(np.ceil(distance / 8.0)))
        for segment in range(0, segments, 2):
            alpha_start = segment / segments
            alpha_end = min(segment + 1, segments) / segments
            point_start = start_point + (end_point - start_point) * alpha_start
            point_end = start_point + (end_point - start_point) * alpha_end
            draw.line(
                [tuple(point_start), tuple(point_end)],
                fill=color,
                width=width,
            )

    for index in range(len(scaled) - 1):
        if position_valid[index] and position_valid[index + 1]:
            if execution_start <= index < execution_stop:
                draw.line(
                    [tuple(scaled[index]), tuple(scaled[index + 1])],
                    fill=phase_color(index),
                    width=4,
                )
            else:
                draw_dashed_line(
                    scaled[index],
                    scaled[index + 1],
                    phase_color(index),
                    width=2,
                )

    for index in range(len(scaled)):
        if not position_valid[index]:
            continue
        x, y = scaled[index]
        color = phase_color(index)
        radius = 7
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline="white",
            width=2,
        )
        if direction_valid[index]:
            draw.line(
                [tuple(scaled[index]), tuple(scaled_direction[index])],
                fill="white",
                width=2,
            )
        draw.text(
            (x + radius + 2, y - radius - 2),
            str(index + 1),
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )

    # The point cloud passed to the model is in the current EEF frame. Its
    # virtual-gripper tail is transformed back to world above and projected
    # here, so this is the exact current gripper input rather than a target-pose
    # approximation. The wireframe makes the four REAP boxes readable even
    # when most surface samples project to the same pixels.
    current_gripper_points_world = np.asarray(
        (
            current_gripper_points_world
            if current_gripper_points_world is not None
            else np.empty((0, 3))
        ),
        dtype=np.float32,
    ).reshape(-1, 3)
    if len(current_gripper_points_world):
        gripper_pixels, gripper_valid = project_world_points_to_front_image(
            current_gripper_points_world, observation
        )
        gripper_pixels = gripper_pixels * scale
        for x, y in gripper_pixels[gripper_valid]:
            draw.ellipse(
                (x - 1.5, y - 1.5, x + 1.5, y + 1.5),
                fill=(0, 235, 255),
            )

    box_edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    valid_box_centers = []
    for box_world in current_gripper_boxes_world or []:
        box_pixels, box_valid = project_world_points_to_front_image(
            np.asarray(box_world, dtype=np.float32), observation
        )
        box_pixels = box_pixels * scale
        for start_index, end_index in box_edges:
            if box_valid[start_index] and box_valid[end_index]:
                segment = [
                    tuple(box_pixels[start_index]),
                    tuple(box_pixels[end_index]),
                ]
                draw.line(segment, fill=(0, 0, 0), width=5)
                draw.line(segment, fill=(0, 235, 255), width=3)
        if np.any(box_valid):
            valid_box_centers.append(box_pixels[box_valid].mean(axis=0))
    if valid_box_centers:
        label_position = np.mean(valid_box_centers, axis=0)
        draw.text(
            (float(label_position[0]) + 6, float(label_position[1]) + 6),
            "current virtual gripper",
            fill=(0, 235, 255),
            stroke_width=2,
            stroke_fill="black",
        )

    draw.rectangle((0, 0, target_width, 62), fill=(0, 0, 0))
    draw.text((8, 5), "Solid path = current execution window", fill="white")
    draw.text((8, 23), "Dashed gray path = future forecast", fill="white")
    draw.text((8, 41), "Cyan wireframe = current virtual gripper", fill=(0, 235, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return position_pixels, position_valid

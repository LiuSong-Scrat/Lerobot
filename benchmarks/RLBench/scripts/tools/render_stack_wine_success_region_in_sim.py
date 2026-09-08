#!/usr/bin/env python3
"""Render one diagnostic stack_wine frame with its success volume visible."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _annotate(frame: np.ndarray, sensor_detected: bool) -> Image.Image:
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    width, height = image.size
    footer_h = 116
    canvas = Image.new("RGB", (width, height + footer_h), (245, 247, 250))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")

    draw.rounded_rectangle(
        (18, 16, 407, 88), radius=14,
        fill=(15, 23, 42, 218), outline=(255, 255, 255, 90), width=2,
    )
    draw.text((36, 27), "RLBench · stack_wine", font=_font(25), fill=(255, 255, 255, 255))
    draw.text((36, 59), "红色透明圆柱 = success 检测体积", font=_font(19), fill=(254, 202, 202, 255))

    status_text = "当前帧：已检测到酒瓶" if sensor_detected else "当前帧：尚未检测到酒瓶"
    status_color = (22, 163, 74, 238) if sensor_detected else (194, 65, 12, 238)
    status_w = 260
    draw.rounded_rectangle(
        (width - status_w - 18, 18, width - 18, 63), radius=12,
        fill=status_color,
    )
    draw.text((width - status_w, 28), status_text, font=_font(18), fill=(255, 255, 255, 255))

    draw.rectangle((0, height, width, height + footer_h), fill=(245, 247, 250, 255))
    draw.ellipse((24, height + 24, 52, height + 52), fill=(239, 68, 68, 180), outline=(153, 27, 27, 255), width=2)
    draw.text((66, height + 18), "半径 2 cm    前向长度 20 cm    检测方向：sensor local +Z（task −Y）",
              font=_font(20), fill=(30, 41, 59, 255))
    draw.text((24, height + 63),
              "这是仿真内加入的无碰撞、不可检测、静态可视化几何；不会改变任务成功条件。",
              font=_font(18), fill=(71, 85, 105, 255))
    return canvas


def render(output: Path, seed: int) -> None:
    # Imports must happen after the launch environment has configured these.
    from pyrep.const import PrimitiveShape
    from pyrep.objects.proximity_sensor import ProximitySensor
    from pyrep.objects.shape import Shape
    from rlbench import CameraConfig, Environment, ObservationConfig
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.tasks.stack_wine import StackWine

    np.random.seed(seed)
    front = CameraConfig(
        rgb=True, depth=False, point_cloud=False, mask=False,
        image_size=(960, 720),
    )
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.front_camera = front
    obs_config.gripper_pose = True
    obs_config.gripper_open = True

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True,
        static_positions=False,
    )
    visual_objects = []
    try:
        env.launch()
        task = env.get_task(StackWine)
        task.set_variation(0)
        task.reset()

        sensor = ProximitySensor("success")

        # A CoppeliaSim cylinder is aligned with its local Z axis. The active
        # proximity volume begins at sensor local z=0 and extends to z=0.20 m,
        # so its visual proxy is centered at local z=0.10 m.
        volume = Shape.create(
            PrimitiveShape.CYLINDER,
            size=[0.040, 0.040, 0.200],
            mass=0.001,
            visible_edges=True,
            smooth=True,
            respondable=False,
            static=True,
            renderable=True,
            color=[1.0, 0.02, 0.02],
        )
        volume.set_position([0.0, 0.0, 0.100], relative_to=sensor)
        volume.set_orientation([0.0, 0.0, 0.0], relative_to=sensor)
        volume.set_collidable(False)
        volume.set_detectable(False)
        volume.set_measurable(False)
        volume.set_transparency(0.58)
        visual_objects.append(volume)

        # Mark the sensor origin in yellow so the cylinder direction is clear.
        origin = Shape.create(
            PrimitiveShape.SPHERE,
            size=[0.012, 0.012, 0.012],
            mass=0.001,
            respondable=False,
            static=True,
            renderable=True,
            color=[1.0, 0.78, 0.02],
        )
        origin.set_position([0.0, 0.0, 0.0], relative_to=sensor)
        origin.set_collidable(False)
        origin.set_detectable(False)
        origin.set_measurable(False)
        visual_objects.append(origin)

        # Let render state settle without executing a robot action.
        for _ in range(3):
            env._pyrep.step()
        observation = task.get_observation()
        frame = np.asarray(observation.front_rgb)
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

        wine_bottle = Shape("wine_bottle")
        detected = bool(sensor.is_detected(wine_bottle))
        annotated = _annotate(frame, detected)
        output.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output, quality=96)
        print(f"output={output}")
        print(f"frame_shape={tuple(frame.shape)}")
        print(f"sensor_detected={detected}")
        print(f"sensor_position_world={sensor.get_position()}")
        print(f"sensor_orientation_world={sensor.get_orientation()}")
    finally:
        for obj in visual_objects:
            try:
                obj.remove()
            except Exception:
                pass
        env.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    render(args.output, args.seed)


if __name__ == "__main__":
    main()

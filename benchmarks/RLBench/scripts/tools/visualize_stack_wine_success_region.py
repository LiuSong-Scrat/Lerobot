#!/usr/bin/env python3
"""Draw the stack_wine success proximity-sensor region for diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Values read from stack_wine.ttm, expressed in the RLBench task-base frame.
SENSOR_X = 0.1101884
SENSOR_Y = 0.3750001
SENSOR_Z = 0.1131164
ACTIVE_Y_NEAR = SENSOR_Y
ACTIVE_Y_FAR = SENSOR_Y - 0.200
SENSOR_RADIUS = 0.020
RACK_CENTER = np.array([0.0501846, 0.2750002, 0.0091195])
RACK_TOP_CENTER = np.array([0.0501885, 0.2750002, 0.1051166])
BOTTLE_RADIUS = 0.029856
BOTTLE_LENGTH = 0.246466


def _set_chinese_font() -> None:
    mpl.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "DejaVu Sans",
    ]
    mpl.rcParams["axes.unicode_minus"] = False


def _cuboid_faces(lo: tuple[float, float, float], hi: tuple[float, float, float]):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    p = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
    )
    return [
        p[[0, 1, 2, 3]], p[[4, 5, 6, 7]], p[[0, 1, 5, 4]],
        p[[2, 3, 7, 6]], p[[1, 2, 6, 5]], p[[0, 3, 7, 4]],
    ]


def _add_cuboid(ax, lo, hi, color, alpha=1.0, edgecolor="#4b5563", linewidth=0.7):
    poly = Poly3DCollection(
        _cuboid_faces(lo, hi), facecolors=color, edgecolors=edgecolor,
        linewidths=linewidth, alpha=alpha,
    )
    ax.add_collection3d(poly)


def _cylinder_along_y(ax, x0, z0, y0, y1, radius, color, alpha, edgecolor=None):
    theta = np.linspace(0, 2 * np.pi, 96)
    ys = np.linspace(y0, y1, 12)
    th, yy = np.meshgrid(theta, ys)
    xx = x0 + radius * np.cos(th)
    zz = z0 + radius * np.sin(th)
    ax.plot_surface(
        xx, yy, zz, color=color, alpha=alpha, linewidth=0, shade=False,
        antialiased=True,
    )
    for y in (y0, y1):
        ax.plot(
            x0 + radius * np.cos(theta), np.full_like(theta, y),
            z0 + radius * np.sin(theta), color=edgecolor or color,
            linewidth=1.3, alpha=min(1.0, alpha + 0.35),
        )


def _draw_3d(ax):
    # A deliberately simplified rack: it is context, not a recovered mesh.
    rack_color = "#a7adb7"
    _add_cuboid(ax, (-0.015, 0.205, 0.015), (0.005, 0.345, 0.170), rack_color, 0.72)
    _add_cuboid(ax, (0.155, 0.205, 0.015), (0.175, 0.345, 0.170), rack_color, 0.72)
    _add_cuboid(ax, (-0.015, 0.205, 0.030), (0.175, 0.345, 0.048), rack_color, 0.72)
    _add_cuboid(ax, (-0.015, 0.205, 0.145), (0.175, 0.345, 0.163), rack_color, 0.72)

    # Bottle envelope, using its measured maximum radius and length.
    bottle_y0 = 0.155
    bottle_y1 = bottle_y0 + BOTTLE_LENGTH
    _cylinder_along_y(
        ax, SENSOR_X, SENSOR_Z, bottle_y0, bottle_y1,
        BOTTLE_RADIUS, "#315b45", 0.17, "#315b45",
    )

    # Exact active sensing cylinder: +local Z maps to task -Y.
    _cylinder_along_y(
        ax, SENSOR_X, SENSOR_Z, ACTIVE_Y_FAR, ACTIVE_Y_NEAR,
        SENSOR_RADIUS, "#ef4444", 0.38, "#b91c1c",
    )
    ax.scatter([SENSOR_X], [SENSOR_Y], [SENSOR_Z], s=42, c="#991b1b", depthshade=False)
    ax.quiver(
        SENSOR_X, SENSOR_Y, SENSOR_Z, 0, -0.075, 0,
        color="#991b1b", linewidth=2.2, arrow_length_ratio=0.20,
    )

    # Task axes.
    origin = np.array([-0.035, 0.405, -0.002])
    ax.quiver(*origin, 0.055, 0, 0, color="#2563eb", linewidth=1.7, arrow_length_ratio=0.18)
    ax.quiver(*origin, 0, -0.055, 0, color="#16a34a", linewidth=1.7, arrow_length_ratio=0.18)
    ax.quiver(*origin, 0, 0, 0.055, color="#d97706", linewidth=1.7, arrow_length_ratio=0.18)
    ax.text(origin[0] + 0.06, origin[1], origin[2], "+X", color="#1d4ed8", fontsize=9)
    ax.text(origin[0], origin[1] - 0.064, origin[2], "−Y", color="#15803d", fontsize=9)
    ax.text(origin[0], origin[1], origin[2] + 0.061, "+Z", color="#b45309", fontsize=9)

    ax.text(SENSOR_X + 0.025, 0.265, SENSOR_Z + 0.035, "success 检测圆柱", color="#991b1b", fontsize=10)
    ax.text(-0.01, 0.31, 0.182, "酒架（简化参照）", color="#4b5563", fontsize=9)
    ax.text(SENSOR_X + 0.033, 0.36, SENSOR_Z - 0.045, "酒瓶外形包络", color="#315b45", fontsize=9)

    ax.set_xlim(-0.05, 0.20)
    ax.set_ylim(0.42, 0.13)  # Reverse so -Y points visually into the rack.
    ax.set_zlim(-0.01, 0.205)
    ax.set_xlabel("task X (m)", labelpad=7)
    ax.set_ylabel("task Y (m)", labelpad=7)
    ax.set_zlabel("task Z (m)", labelpad=7)
    ax.set_box_aspect((0.25, 0.29, 0.215))
    ax.view_init(elev=23, azim=-55)
    ax.grid(True, alpha=0.18)
    ax.set_title("三维位置关系（红色区域按真实尺寸）", fontsize=13, pad=12, weight="bold")


def _draw_side_view(ax):
    ax.set_title("侧视图：Y–Z", fontsize=12, weight="bold", loc="left")
    # Active volume appears as a rectangle in axial cross-section.
    rect = Rectangle(
        (ACTIVE_Y_FAR, SENSOR_Z - SENSOR_RADIUS),
        ACTIVE_Y_NEAR - ACTIVE_Y_FAR,
        2 * SENSOR_RADIUS,
        facecolor="#ef4444", edgecolor="#b91c1c", alpha=0.35, linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.axvline(RACK_CENTER[1], color="#6b7280", linestyle="--", linewidth=1.2)
    ax.scatter([SENSOR_Y], [SENSOR_Z], c="#991b1b", s=35, zorder=5)
    ax.annotate(
        "sensor 原点\n(y=0.3750 m)", xy=(SENSOR_Y, SENSOR_Z),
        xytext=(0.392, 0.159), fontsize=9, ha="center",
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.1),
    )
    ax.annotate(
        "", xy=(ACTIVE_Y_FAR, SENSOR_Z - 0.036),
        xytext=(ACTIVE_Y_NEAR, SENSOR_Z - 0.036),
        arrowprops=dict(arrowstyle="<->", color="#111827", lw=1.4),
    )
    ax.text(
        (ACTIVE_Y_FAR + ACTIVE_Y_NEAR) / 2, SENSOR_Z - 0.044,
        "有效前向长度 20 cm", ha="center", va="top", fontsize=10,
    )
    ax.text(RACK_CENTER[1], 0.164, "酒架中心 y≈0.275 m", rotation=90,
            va="top", ha="right", fontsize=8.5, color="#4b5563")
    ax.text(0.260, SENSOR_Z + 0.006, "检测方向  −Y", fontsize=10, color="#991b1b")
    ax.set_xlim(0.145, 0.415)
    ax.set_ylim(0.058, 0.176)
    ax.set_xlabel("task Y (m)")
    ax.set_ylabel("task Z (m)")
    ax.grid(True, alpha=0.22)
    ax.set_aspect("equal", adjustable="box")


def _draw_cross_section(ax):
    ax.set_title("横截面：X–Z（任意 Y∈[0.175, 0.375] m）", fontsize=12, weight="bold", loc="left")
    sensor = Circle(
        (SENSOR_X, SENSOR_Z), SENSOR_RADIUS,
        facecolor="#ef4444", edgecolor="#b91c1c", alpha=0.38, linewidth=1.8,
    )
    bottle = Circle(
        (SENSOR_X, SENSOR_Z), BOTTLE_RADIUS,
        facecolor="none", edgecolor="#315b45", linestyle="--", linewidth=1.7,
    )
    ax.add_patch(bottle)
    ax.add_patch(sensor)
    ax.scatter([SENSOR_X], [SENSOR_Z], c="#991b1b", s=24, zorder=5)
    ax.annotate(
        "", xy=(SENSOR_X - SENSOR_RADIUS, SENSOR_Z - 0.038),
        xytext=(SENSOR_X + SENSOR_RADIUS, SENSOR_Z - 0.038),
        arrowprops=dict(arrowstyle="<->", color="#111827", lw=1.4),
    )
    ax.text(SENSOR_X, SENSOR_Z - 0.045, "直径 4 cm（半径 2 cm）", ha="center", va="top", fontsize=10)
    ax.annotate(
        "圆心 (x,z)=(0.1102, 0.1131) m",
        xy=(SENSOR_X, SENSOR_Z), xytext=(0.145, 0.157), fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.1),
    )
    ax.text(0.066, 0.153, "虚线：酒瓶最大外径约 5.97 cm", fontsize=8.8, color="#315b45")
    ax.set_xlim(0.055, 0.175)
    ax.set_ylim(0.052, 0.178)
    ax.set_xlabel("task X (m)")
    ax.set_ylabel("task Z (m)")
    ax.grid(True, alpha=0.22)
    ax.set_aspect("equal", adjustable="box")


def make_figure(output_path: Path) -> None:
    _set_chinese_font()
    fig = plt.figure(figsize=(16, 9.5), dpi=180, facecolor="#f8fafc")
    gs = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0), height_ratios=(1, 1),
                          left=0.045, right=0.975, top=0.88, bottom=0.12,
                          wspace=0.20, hspace=0.34)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d", facecolor="#f8fafc")
    ax_side = fig.add_subplot(gs[0, 1], facecolor="white")
    ax_cross = fig.add_subplot(gs[1, 1], facecolor="white")

    _draw_3d(ax3d)
    _draw_side_view(ax_side)
    _draw_cross_section(ax_cross)

    fig.suptitle("RLBench · stack_wine 成功判定范围", fontsize=21, weight="bold", y=0.965)
    fig.text(
        0.5, 0.918,
        "success = wine_bottle 的可检测表面进入 success ProximitySensor 圆柱体",
        ha="center", fontsize=12.5, color="#334155",
    )
    fig.text(
        0.055, 0.045,
        "判定只要求传感器检测到 wine_bottle；不额外要求松爪、静止或指定姿态。"
        "  注：酒架采用简化形状，仅作位置参照；红色圆柱的坐标与尺寸来自 stack_wine.ttm。",
        fontsize=10.5, color="#334155",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#eef2ff", edgecolor="#c7d2fe"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()

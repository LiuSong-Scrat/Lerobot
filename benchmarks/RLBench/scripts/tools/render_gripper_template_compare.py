#!/usr/bin/env python3
"""Render the RLBench and LIBERO two-finger point-cloud templates side by side."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _rlbench_tool_paths import LEROBOT_ROOT as ROOT
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "song_real_libero" / "scripts"))

from libero_setting.libero_pointcloud_utils import (  # noqa: E402
    create_rlbench_panda_gripper_points,
)
from rlbench_reap_gripper import (  # noqa: E402
    RLBENCH_PANDA_PHYSICAL_MAX_WIDTH,
    create_rlbench_reap_points_from_physical_width,
)


def main() -> None:
    output = ROOT / "benchmarks" / "RLBench" / "outputs" / "gripper_template_compare.png"
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    pose = np.zeros(6, dtype=np.float32)
    # Compare both templates at RLBench's same physical 0.08 m aperture.
    rlbench = create_rlbench_panda_gripper_points(1.0, pose, 500, rng_a)
    libero = create_rlbench_reap_points_from_physical_width(
        RLBENCH_PANDA_PHYSICAL_MAX_WIDTH, pose, 500, rng_b
    )

    fig = plt.figure(figsize=(14, 7), dpi=160)
    for index, (title, points, color) in enumerate(
        (("RLBench Panda measured template", rlbench, "#1677ff"),
         ("RLBench-aligned REAP template v4", libero, "#e05a33")),
        start=1,
    ):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c=color, alpha=0.8)
        ax.scatter([0], [0], [0], s=35, c="black", marker="x")
        ax.set_title(title)
        ax.set_xlabel("EEF x (m)")
        ax.set_ylabel("EEF y (m)")
        ax.set_zlabel("EEF z (m)")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-58)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Open gripper point clouds at the same EEF pose; black x = EEF origin")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()

from pyrep.backend import sim
from pyrep.robots.configuration_paths.configuration_path import (
    ConfigurationPath)
import numpy as np
from typing import List, Union


class ArmConfigurationPath(ConfigurationPath):
    """A path expressed in joint configuration space.

    Paths are retrieved from an :py:class:`Arm`, and are associated with the
    arm that generated the path.

    This class is used for executing motion along a path via the
    Reflexxes Motion Library type II or IV. The Reflexxes Motion Library
    provides instantaneous trajectory generation capabilities for motion
    control systems.
    """

    def __init__(self, arm: 'Arm',  # type: ignore
                 path_points: Union[List[float], np.ndarray]):
        self._arm = arm
        self._path_points = np.asarray(path_points)
        self._drawing_handle = None
        self._path_done = False
        self._num_joints = arm.get_joint_count()
        self._joint_position_action = None
        self._path_configs = self._path_points.reshape(-1, self._num_joints)
        self._path_index = 0
        self._position_tolerance = 1e-3

    def __len__(self):
        return len(self._path_points) // self._num_joints

    def __getitem__(self, i):
        path_points = self._path_configs[i].flatten()
        return self.__class__(arm=self._arm, path_points=path_points)

    def step(self) -> bool:
        """Makes a step along the trajectory.

        This function steps forward a trajectory generation algorithm from
        Reflexxes Motion Library.
        NOTE: This does not step the physics engine. This is left to the user.

        :return: If the end of the trajectory has been reached.
        """
        if self._path_done:
            raise RuntimeError('This path has already been completed. '
                               'If you want to re-run, then call set_to_start.')
        # CoppeliaSim 4.10 removed the legacy simRML* functions.  The simIK
        # path already contains the complete joint trajectory.  Applying one
        # generated configuration per physics step preserves the old
        # path.step()/scene.step() contract and avoids waiting indefinitely
        # for a legacy position servo to converge.
        target = np.asarray(self._path_configs[self._path_index], dtype=np.float64)
        self._joint_position_action = target.copy()
        self._arm.set_joint_positions(target.tolist())
        self._path_done = self._path_index >= len(self._path_configs) - 1
        if not self._path_done:
            self._path_index += 1
        return self._path_done

    def set_to_start(self, disable_dynamics=False) -> None:
        """Sets the arm to the beginning of this path.
        """
        start_config = self._path_points[:len(self._arm.joints)]
        self._arm.set_joint_positions(start_config, disable_dynamics=disable_dynamics)
        self._path_done = False
        self._path_index = 0
        self._joint_position_action = None

    def set_to_end(self, disable_dynamics=False) -> None:
        """Sets the arm to the end of this path.
        """
        final_config = self._path_points[-len(self._arm.joints):]
        self._arm.set_joint_positions(final_config, disable_dynamics=disable_dynamics)
        self._path_index = len(self._path_configs) - 1
        self._path_done = True

    def visualize(self) -> None:
        """Draws a visualization of the path in the scene.

        The visualization can be removed
        with :py:meth:`ConfigurationPath.clear_visualization`.
        """
        if len(self._path_points) <= 0:
            raise RuntimeError("Can't visualise a path with no points.")

        tip = self._arm.get_tip()
        self._drawing_handle = sim.simAddDrawingObject(
            objectType=sim.sim_drawing_lines, size=3, duplicateTolerance=0,
            parentObjectHandle=-1, maxItemCount=99999,
            ambient_diffuse=[1, 0, 1])
        sim.simAddDrawingObjectItem(self._drawing_handle, None)
        init_angles = self._arm.get_joint_positions()
        self._arm.set_joint_positions(
            self._path_points[0: len(self._arm.joints)])
        prev_point = list(tip.get_position())

        for i in range(len(self._arm.joints), len(self._path_points),
                       len(self._arm.joints)):
            points = self._path_points[i:i + len(self._arm.joints)]
            self._arm.set_joint_positions(points)
            p = list(tip.get_position())
            sim.simAddDrawingObjectItem(self._drawing_handle, prev_point + p)
            prev_point = p

        # Set the arm back to the initial config
        self._arm.set_joint_positions(init_angles)

    def clear_visualization(self) -> None:
        """Clears/removes a visualization of the path in the scene.
        """
        if self._drawing_handle is not None:
            sim.simAddDrawingObjectItem(self._drawing_handle, None)

    def get_executed_joint_position_action(self) -> np.ndarray:
        return self._joint_position_action

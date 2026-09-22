# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Planned Cartesian retract along a straight-line direction.

``safe_retract()`` moves the arm's end-effector along a twist direction
up to a maximum distance. The trajectory is planned collision-free
(IK + collision check at every waypoint via ``plan_cartesian_path``).

Runtime collision detection is delegated to the hardware safety layer
(e.g. UR protective stop) rather than software contact monitoring.
The only software abort is via ``stop_condition`` (e-stop, ownership).

Use for:

- Post-grasp lift
- Recovery retract after failed grasp
- Any directional motion away from a surface
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from mj_manipulator.arm_group import ArmGroup
    from mj_manipulator.protocols import ExecutionContext

logger = logging.getLogger(__name__)


def safe_retract(
    arm_group: ArmGroup,
    ctx: ExecutionContext,
    twist: np.ndarray,
    max_distance: float,
    *,
    segment_length: float = 0.005,
    max_branch_jump: float | None = None,
    stop_condition: Callable[[], bool] | None = None,
) -> float:
    """Move all EEs along ``twist`` up to ``max_distance``.

    Plans a collision-free Cartesian trajectory for every arm in the group
    along the twist direction and executes it via ``ctx.execute``. The
    trajectory is planned with collision checking at every waypoint; runtime
    collision protection is handled by the hardware safety layer.

    Currently handles translational twists only (angular components
    ``twist[3:]`` must be zero).

    Args:
        arm_group: Arm group to move. Every arm must have an IK solver attached.
        ctx: Execution context (SimContext or HardwareContext).
        twist: 6D twist [vx, vy, vz, wx, wy, wz]. The translational part is
            applied to every arm; angular components must be zero.
        max_distance: Maximum distance to travel along the twist (meters).
        segment_length: Cartesian spacing between IK waypoints (meters).
        max_branch_jump: Maximum per-waypoint joint-space step (radians,
            vector norm across all joints). ``None`` disables the check.
        stop_condition: Optional early-termination predicate (e-stop,
            ownership abort). Checked each control cycle.

    Returns:
        Signed projected distance traveled along the twist direction (meters).
    """
    if not np.allclose(twist[3:], 0.0):
        raise NotImplementedError(
            f"safe_retract currently handles translational twists only; "
            f"got angular components {twist[3:]}"
        )

    linear = np.asarray(twist[:3], dtype=float)
    linear_norm = float(np.linalg.norm(linear))
    if linear_norm < 1e-9:
        logger.warning("safe_retract: twist has zero magnitude; nothing to do")
        return 0.0

    direction = linear / linear_norm

    start_poses = arm_group.get_ee_pose()
    start_positions = [
        pose[:3, 3].copy()
        for pose in start_poses
    ]

    goals = {}

    for arm_name, start_pose in zip(arm_group.arms, start_poses):
        goals[arm_name] = translational_waypoints(
            start_pose,
            direction,
            max_distance,
            segment_length=segment_length,
        )

    try:
        trajectory = arm_group.plan_cartesian_path(
            goals,
            max_branch_jump=max_branch_jump,
            partial_ok=True,
        )
    except ValueError as exc:
        logger.warning("safe_retract: no feasible prefix (%s); not moving", exc)
        return 0.0

    ctx.execute(trajectory, abort_fn=stop_condition)

    end_poses = arm_group.get_ee_pose()
    distances_traveled = [
        float(np.dot(end_pose[:3, 3] - start_pos, direction))
        for end_pose, start_pos in zip(end_poses, start_positions)
    ]

    distance_traveled = min(distances_traveled)

    logger.info(
        "safe_retract: moved %.3fm along twist (arms: %s)",
        distance_traveled,
        ", ".join(
            f"{name}={distance:.3f}m"
            for name, distance in zip(arm_group.arms, distances_traveled)
        ),
    )

    return distance_traveled


def translational_waypoints(
    start_pose: np.ndarray,
    direction: np.ndarray,
    distance: float,
    *,
    segment_length: float = 0.005,
) -> list[np.ndarray]:
    """Generate SE(3) waypoints along a straight-line translation.

    Produces waypoints at ``segment_length`` increments from ``start_pose``
    in the given world-frame ``direction``, preserving orientation. The
    final waypoint is exactly at ``start + direction * distance`` (possibly
    closer than ``segment_length`` to the penultimate waypoint).

    Useful for :func:`plan_cartesian_path` callers who want a simple
    straight-line Cartesian motion (post-grasp lift, pre-grasp approach,
    post-place retreat).

    Args:
        start_pose: 4x4 SE(3) starting pose in world frame.
        direction: 3D direction vector. Will be normalized internally.
        distance: Total translation distance along ``direction`` (meters).
        segment_length: Spacing between consecutive waypoints (meters).
            Smaller = more IK solves but smoother path reconstruction.
            5 mm is a good default for manipulation-scale motions.

    Returns:
        List of 4x4 SE(3) waypoint poses.
    """
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return []
    unit = direction / norm

    start_rot = np.asarray(start_pose[:3, :3], dtype=float)
    start_trans = np.asarray(start_pose[:3, 3], dtype=float)

    n_segments = max(1, int(np.ceil(distance / segment_length)))
    waypoints: list[np.ndarray] = []
    for i in range(1, n_segments + 1):
        step = min(i * segment_length, distance)
        pose = np.eye(4)
        pose[:3, :3] = start_rot
        pose[:3, 3] = start_trans + unit * step
        waypoints.append(pose)
    return waypoints


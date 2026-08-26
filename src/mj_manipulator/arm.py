# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generic robot arm abstraction for MuJoCo manipulators.

Wraps an Environment + ArmConfig to provide:
- State queries (joint positions, EE pose, joint limits)
- Forward kinematics (non-destructive, for planning)
- Motion planning via pycbirrt (config-to-config, TSR-based, pose-based)

Robot-specific code (IK solvers, grippers) is injected via protocols.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mj_manipulator.config import ArmConfig

if TYPE_CHECKING:
    from mj_environment import Environment

    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.protocols import Gripper, IKSolver

logger = logging.getLogger(__name__)


# =============================================================================
# pycbirrt RobotModel adapters
# =============================================================================


class ArmRobotModel:
    """Adapts Arm for pycbirrt's RobotModel protocol (single-threaded).

    Uses Arm.forward_kinematics() which creates a temporary MjData copy,
    so it's safe for planning but not thread-safe.
    """

    def __init__(self, arm: Arm):
        self._arm = arm

    @property
    def dof(self) -> int:
        return self._arm.dof

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._arm.get_joint_limits()

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        return self._arm.forward_kinematics(q)


# =============================================================================
# Helpers
# =============================================================================


def _read_site_pose(
    data: mujoco.MjData,
    site_id: int,
    tcp_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Read a 4x4 pose from a MuJoCo site, optionally applying tcp_offset."""
    pos = data.site_xpos[site_id]
    mat = data.site_xmat[site_id].reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = mat
    T[:3, 3] = pos
    if tcp_offset is not None:
        T = T @ tcp_offset
    return T


def add_subtree_gravcomp(
    spec: mujoco.MjSpec,
    root_body_name: str,
) -> int:
    """Enable gravity compensation on a body and all its descendants.

    Must be called **before** ``spec.compile()``. MuJoCo optimizes
    gravcomp away at compile time if every body has ``gravcomp=0``;
    runtime writes to ``model.body_gravcomp`` are silently ignored.
    That's why this helper operates on the MjSpec (editable) rather
    than a compiled ``MjModel``.

    Walks the MjSpec body tree rooted at ``root_body_name`` via an
    explicit stack (robust to whatever iteration the MjSpec API
    provides) and sets ``body.gravcomp = 1.0`` on every body it
    finds. This is the primitive that per-arm helpers like
    :func:`mj_manipulator.arms.franka.add_franka_gravcomp` delegate
    to — they just know the right root body name.

    Idempotent: calling twice on the same spec, or on overlapping
    subtrees, is harmless because setting ``gravcomp = 1.0`` twice
    produces the same result.

    Failure modes handled:

    - **root_body_name not found**: raises ``ValueError`` with the
      bad name AND the list of top-level world-body children, so
      typos are easy to diagnose.
    - **Root with no descendants**: still sets gravcomp on the root
      and returns ``count = 1``. Degenerate but valid.

    Not handled (caller's responsibility):

    - Calling on an already-compiled spec. The MjSpec API doesn't
      expose a reliable "was this compiled?" check; the call will
      still "succeed" but have no effect on the existing MjModel.
    - Scoping: if you pass a root that's an ancestor of bodies you
      don't want gravcomp'd (e.g. the world body, or a linear base
      beneath the arm), the walker will touch them too. Per-arm
      helpers sidestep this by passing the arm's kinematic root,
      not a higher ancestor.

    Args:
        spec: MjSpec loaded from a scene XML. Must not have been
            compiled yet (or the gravcomp change will be ignored).
        root_body_name: Name of the root body for the gravcomp
            subtree. Typically the arm's base link (e.g. ``"link0"``
            for Franka, ``"base"`` for UR5e).

    Returns:
        Number of bodies that had ``gravcomp`` set. Useful for
        sanity-checking that the walker touched the expected count
        (11 for Franka, 7 for bare UR5e, etc.).

    Raises:
        ValueError: If ``root_body_name`` is not found in the spec.
            The message includes the bad name and the list of
            top-level children of ``spec.worldbody`` so the caller
            can see what's actually there.
    """
    root = spec.body(root_body_name)
    if root is None:
        available = [b.name for b in spec.worldbody.bodies if b.name]
        raise ValueError(
            f"add_subtree_gravcomp: body '{root_body_name}' not found in spec. "
            f"Top-level worldbody children: {available}"
        )

    count = 0
    stack = [root]
    while stack:
        body = stack.pop()
        body.gravcomp = 1.0
        count += 1
        stack.extend(body.bodies)
    return count


# =============================================================================
# Arm
# =============================================================================


class Arm:
    """Generic robot arm abstraction.

    Provides state queries, forward kinematics, and motion planning for
    any MuJoCo robot arm. Robot-specific capabilities (IK, gripper) are
    injected via protocols.

    Args:
        env: MuJoCo environment (provides model and data).
        config: Arm configuration (joint names, limits, ee_site, etc.).
        gripper: Optional gripper implementation.
        grasp_manager: Optional grasp state tracker.
        ik_solver: Optional IK solver for pose-based planning.
    """

    env: Environment
    config: ArmConfig
    gripper: Gripper | None
    grasp_manager: GraspManager | None
    ik_solver: IKSolver | None
    joint_ids: list[int]
    joint_qpos_indices: list[int]
    joint_qvel_indices: list[int]
    ee_site_id: int
    dof: int

    def __init__(
        self,
        env: Environment,
        config: ArmConfig,
        *,
        gripper: Gripper | None = None,
        grasp_manager: GraspManager | None = None,
        ik_solver: IKSolver | None = None,
    ):
        self.env: Environment = env
        self.config: ArmConfig = config
        self.gripper: Gripper | None = gripper
        self.grasp_manager: GraspManager | None = grasp_manager
        self.ik_solver: IKSolver | None = ik_solver
        self.ft_valid: bool = False  # Set by ExecutionContext when F/T is meaningful
        self._ft_tare_offset: np.ndarray = np.zeros(6)  # Tare baseline

        model = env.model

        # Resolve joint IDs and cache indices
        self.joint_ids: list[int] = []
        self.joint_qpos_indices: list[int] = []
        self.joint_qvel_indices: list[int] = []

        for name in config.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found in model")
            self.joint_ids.append(jid)
            self.joint_qpos_indices.append(model.jnt_qposadr[jid])
            self.joint_qvel_indices.append(model.jnt_dofadr[jid])

        # Resolve EE site
        self.ee_site_id: int
        if config.ee_site:
            self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config.ee_site)
            if self.ee_site_id == -1:
                raise ValueError(f"EE site '{config.ee_site}' not found in model")
        else:
            self.ee_site_id = -1

        # Resolve actuator IDs (actuators whose transmission targets our joints)
        # Filter by trntype to exclude tendon/site actuators whose trnid
        # could collide with joint IDs (e.g. Franka gripper tendon actuator).
        self.actuator_ids: list[int] = []
        joint_id_set = set(self.joint_ids)
        for act_id in range(model.nu):
            trntype = model.actuator_trntype[act_id]
            if trntype == mujoco.mjtTrn.mjTRN_JOINT and (model.actuator_trnid[act_id, 0] in joint_id_set):
                self.actuator_ids.append(act_id)

        # Cache DOF and joint limits
        self.dof: int = len(config.joint_names)
        self._joint_limits: tuple[np.ndarray, np.ndarray] | None = None

        # Check whether gravity compensation is active on the arm subtree.
        # MuJoCo optimizes gravcomp away at compile time if every body has
        # gravcomp=0, and runtime changes to body_gravcomp are silently
        # ignored. The per-arm MjSpec helpers (add_franka_gravcomp,
        # add_ur5e_gravcomp) must be called BEFORE spec.compile(). Real
        # robot controllers (Franka FCI, UR RTDE, KUKA FRI, etc.) run
        # gravity compensation internally; without it in sim, the PD loop
        # must fight gravity via steady-state position error, producing
        # sag at rest and tracking lag in motion.
        first_joint_body = model.jnt_bodyid[self.joint_ids[0]]
        base_body_id = model.body_parentid[first_joint_body]
        subtree_has_gravcomp = any(model.body_gravcomp[bid] > 0 for bid in range(base_body_id, model.nbody))
        if not subtree_has_gravcomp:
            logger.warning(
                "Arm '%s' has no gravity compensation on its kinematic "
                "subtree. Call add_%s_gravcomp(spec) BEFORE spec.compile() "
                "(or bake gravcomp='1' into the source XML). Without it, "
                "the PD loop must fight gravity via steady-state position "
                "error, producing sag at rest and tracking lag in motion.",
                config.name,
                config.name,
            )

        # Resolve F/T sensor indices (if configured)
        self._ft_force_adr: int | None = None
        self._ft_torque_adr: int | None = None
        self.ft_site_id: int | None = None
        if config.ft_force_sensor:
            sid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                config.ft_force_sensor,
            )
            if sid == -1:
                raise ValueError(f"Force sensor '{config.ft_force_sensor}' not found")
            self._ft_force_adr = model.sensor_adr[sid]
            self.ft_site_id = model.sensor_objid[sid]
        if config.ft_torque_sensor:
            sid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                config.ft_torque_sensor,
            )
            if sid == -1:
                raise ValueError(f"Torque sensor '{config.ft_torque_sensor}' not found")
            self._ft_torque_adr = model.sensor_adr[sid]

    # -----------------------------------------------------------------
    # State queries
    # -----------------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        """Current joint positions (rad)."""
        return np.array([self.env.data.qpos[idx] for idx in self.joint_qpos_indices])

    def set_joint_positions(self, q: np.ndarray, ctx=None) -> None:
        """Set joint positions directly, sync viewer.

        Simulation only — teleports the arm to the target configuration.
        On real hardware, use plan_to_configuration() instead.

        Args:
            q: Joint positions (rad), length must match DOF.
            ctx: ExecutionContext for syncing. If None, runs mj_forward only.
        """
        q = np.asarray(q, dtype=float)
        if len(q) != self.dof:
            raise ValueError(f"Expected {self.dof} joints, got {len(q)}")
        lower, upper = self.get_joint_limits()
        model = self.env.model
        for i in range(self.dof):
            jid = self.joint_ids[i]
            if model.jnt_limited[jid] and (q[i] < lower[i] or q[i] > upper[i]):
                raise ValueError(f"Joint {i} value {q[i]:.3f} outside limits [{lower[i]:.3f}, {upper[i]:.3f}]")
        for i, idx in enumerate(self.joint_qpos_indices):
            self.env.data.qpos[idx] = q[i]
        for idx in self.joint_qvel_indices:
            self.env.data.qvel[idx] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        if ctx is not None:
            ctx.sync()

    def get_joint_velocities(self) -> np.ndarray:
        """Current joint velocities (rad/s)."""
        return np.array([self.env.data.qvel[idx] for idx in self.joint_qvel_indices])

    def get_joint_torques(self) -> np.ndarray:
        """Current joint-torque vector (N·m per joint).

        Returns ``qfrc_actuator`` for this arm's joints — the torque each
        actuator applies to the joint. With gravity compensation active,
        this reduces (at rest) largely to whatever external load the arm
        is working against: the weight of a held object, contact forces,
        etc. That makes it a useful load signal for arms whose only load
        sensing is at the joints (e.g. Franka via ``tau_ext``), parallel
        to :meth:`get_ft_wrench` for arms with a wrist F/T sensor.

        Returns NaN when ``ft_valid`` is False — the same validity gate
        as F/T. Kinematic sim doesn't run physics integration, so
        ``qfrc_actuator`` is meaningless there. On real hardware the
        ``HardwareContext`` supplies the driver's external-torque
        estimate and sets ``ft_valid = True``.

        Returns:
            np.ndarray of shape ``(dof,)``: actuator torques in N·m,
            indexed to match ``get_joint_positions()``. All NaN if joint
            torque data is not currently meaningful.
        """
        if not self.ft_valid:
            return np.full(self.dof, np.nan)
        data = self.env.data
        return np.array([data.qfrc_actuator[idx] for idx in self.joint_qvel_indices])

    def get_ft_wrench(self) -> np.ndarray:
        """Current wrist force/torque reading as [fx, fy, fz, tx, ty, tz].

        Returns the 6D wrench from the wrist F/T sensor in the **sensor
        local frame** (not world frame). The sensor reports the force
        exerted on the child body (gripper) by the parent body (wrist).

        Returns NaN when ``ft_valid`` is False (default). The execution
        context sets ``ft_valid = True`` when F/T data is meaningful:
        physics sim after ``mj_step``, or real hardware with a live
        sensor. In kinematic sim, MuJoCo's constraint solver produces
        artifact values (100-300N) that are not physical wrist forces.

        To transform to world frame::

            wrench = arm.get_ft_wrench()
            R = data.site_xmat[arm.ft_site_id].reshape(3, 3)
            force_world = R @ wrench[:3]
            torque_world = R @ wrench[3:]

        Returns:
            np.ndarray of shape (6,): [fx, fy, fz, tx, ty, tz] in sensor frame.
            All NaN if no physics step has been run.

        Raises:
            RuntimeError: If no F/T sensor is configured.
        """
        if self._ft_force_adr is None or self._ft_torque_adr is None:
            raise RuntimeError("No F/T sensor configured. Set ft_force_sensor and ft_torque_sensor in ArmConfig.")
        if not self.ft_valid:
            return np.full(6, np.nan)
        data = self.env.data
        force = data.sensordata[self._ft_force_adr : self._ft_force_adr + 3]
        torque = data.sensordata[self._ft_torque_adr : self._ft_torque_adr + 3]
        return np.concatenate([force, torque]) - self._ft_tare_offset

    def get_ft_wrench_world(self) -> np.ndarray:
        """Current wrist force/torque reading in the **world frame**.

        Convenience wrapper around :meth:`get_ft_wrench` that rotates
        the wrench from the sensor local frame to the world frame.

        Returns:
            np.ndarray of shape (6,): [fx, fy, fz, tx, ty, tz] in world frame.
            All NaN if F/T is not valid (kinematic mode).
        """
        wrench = self.get_ft_wrench()
        if np.isnan(wrench[0]):
            return wrench
        R = self.env.data.site_xmat[self.ft_site_id].reshape(3, 3)
        force_world = R @ wrench[:3]
        torque_world = R @ wrench[3:]
        return np.concatenate([force_world, torque_world])

    def tare_ft(self) -> None:
        """Zero the F/T sensor at the current reading (tare).

        Records the current raw sensor reading as the baseline. All
        subsequent ``get_ft_wrench()`` calls return the delta from
        this baseline. Arm should be stationary when taring.

        Matches UR5e's ``zero_ftsensor()`` URScript command.
        """
        if self._ft_force_adr is None or self._ft_torque_adr is None:
            raise RuntimeError("No F/T sensor configured.")
        if not self.ft_valid:
            raise RuntimeError("F/T not valid (kinematic mode). Use physics mode to tare.")
        data = self.env.data
        force = data.sensordata[self._ft_force_adr : self._ft_force_adr + 3]
        torque = data.sensordata[self._ft_torque_adr : self._ft_torque_adr + 3]
        self._ft_tare_offset = np.concatenate([force, torque]).copy()

    @property
    def has_ft_sensor(self) -> bool:
        """Whether this arm has a wrist F/T sensor configured."""
        return self._ft_force_adr is not None and self._ft_torque_adr is not None

    def get_ee_pose(self) -> np.ndarray:
        """Current end-effector pose as 4x4 homogeneous transform.

        Calls mj_forward to ensure kinematics are up-to-date, then reads
        the EE site pose. Applies tcp_offset if configured.
        """
        if self.ee_site_id == -1:
            raise RuntimeError("No ee_site configured")
        mujoco.mj_forward(self.env.model, self.env.data)
        return _read_site_pose(self.env.data, self.ee_site_id, self.config.tcp_offset)

    def get_ee_jacobian(self) -> np.ndarray:
        """6xN end-effector Jacobian in world frame.

        Rows 0-2 are linear velocity, rows 3-5 are angular velocity.
        Columns correspond to arm joints in joint_qvel_indices order.
        """
        if self.ee_site_id == -1:
            raise RuntimeError("No ee_site configured")
        from mj_manipulator.cartesian import get_ee_jacobian

        return get_ee_jacobian(self.env.model, self.env.data, self.ee_site_id, self.joint_qvel_indices)

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint position limits as (lower, upper) arrays.

        Unlimited joints (limited=False) return [-π, π] as a nominal
        range for sampling/planning. The actual joint has no physical
        limit — angular_joints handles wrapping in the planner.
        """
        if self._joint_limits is None:
            model = self.env.model
            lower = []
            upper = []
            for jid in self.joint_ids:
                if model.jnt_limited[jid]:
                    lower.append(model.jnt_range[jid, 0])
                    upper.append(model.jnt_range[jid, 1])
                else:
                    # Unlimited joint — use ±2π as nominal range.
                    # The planner's angular_joints handles wrapping.
                    lower.append(-2 * np.pi)
                    upper.append(2 * np.pi)
            self._joint_limits = (np.array(lower), np.array(upper))
        return self._joint_limits

    # -----------------------------------------------------------------
    # Forward kinematics (non-destructive, for planning)
    # -----------------------------------------------------------------

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute EE pose at configuration q without modifying live state.

        Creates a temporary MjData copy, sets joints to q, runs mj_forward,
        and reads the resulting pose. The live env.data is never touched.
        """
        if self.ee_site_id == -1:
            raise RuntimeError("No ee_site configured")

        tmp_data = mujoco.MjData(self.env.model)
        # Copy current state as baseline
        np.copyto(tmp_data.qpos, self.env.data.qpos)
        # Set arm joints to requested config
        for i, idx in enumerate(self.joint_qpos_indices):
            tmp_data.qpos[idx] = q[i]
        mujoco.mj_forward(self.env.model, tmp_data)
        return _read_site_pose(tmp_data, self.ee_site_id, self.config.tcp_offset)

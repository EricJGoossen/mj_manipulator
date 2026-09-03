from __future__ import annotations

import dataclasses
import itertools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING
from collections.abc import Iterator, Mapping

import mujoco
import numpy as np
from pycbirrt import CBiRRT, CBiRRTConfig
from pycbirrt.exceptions import PlanningError
from tsr import TSR

from mj_manipulator.collision import CollisionChecker
from mj_manipulator.config import ArmGroupConfig, KinematicLimits
from mj_manipulator.trajectory import Trajectory
from mj_manipulator.arm import read_site_pose as read_single_site_pose

if TYPE_CHECKING:
    from mj_environment import Environment

    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.arm import Arm

logger = logging.getLogger(__name__)

class ContextRobotModel:
    """Thread-safe RobotModel adapter using isolated MjData.

    Each instance owns a private MjData copy for FK computation.
    Created by Arm.create_planner() for parallel planning.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_qpos_indices: list[int],
        ee_site_id_group: list[int],
        joint_limits: tuple[np.ndarray, np.ndarray],    
        tcp_offset_group: list[np.ndarray] | None = None,
    ):
        self._model = model
        self._data = data
        self._joint_qpos_indices = joint_qpos_indices
        self.ee_site_id_group = ee_site_id_group
        self._joint_limits = joint_limits
        self.tcp_offset_group = tcp_offset_group

    @property
    def dof(self) -> int:
        return len(self._joint_qpos_indices)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._joint_limits

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute EE pose on private data (thread-safe)."""
        for i, idx in enumerate(self._joint_qpos_indices):
            self._data.qpos[idx] = q[i]
        mujoco.mj_forward(self._model, self._data)
        poses = _read_site_pose(self._data, self.ee_site_id_group, self.tcp_offset_group)
        return poses[0] if len(poses) == 1 else poses

# =============================================================================
# Helpers
# =============================================================================


def _read_site_pose(
    data: mujoco.MjData,
    ee_site_id_group: list[int],
    tcp_offset_group: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Read a 4x4 pose from a MuJoCo site, optionally applying tcp_offset."""
    return [
        read_single_site_pose(
            data, 
            site_id, 
            tcp_offset_group[i] if tcp_offset_group is not None else None
        )
        for i, site_id in enumerate(ee_site_id_group)
    ]

# =============================================================================
# Arm group
# =============================================================================

class ArmGroup(Mapping):
    arms: dict[str, Arm]
    dof: int
    config: ArmGroupConfig
    env: Environment
    grasp_manager: GraspManager | None

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def __init__(
        self,
        arms: dict[str, Arm],
        config: ArmGroupConfig,
    ):
        self.arms = arms
        self.config = config

        self.dof = sum(arm.dof for arm in arms.values())
        if list(self.joint_names) != list(config.joint_names):
            raise ValueError(
                "ArmGroup joint_names mismatch: "
                f"config={config.joint_names}, arms={self.joint_names}"
            )

    def _split_by_arm(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Split a joint position array into per-arm arrays."""
        q_by_arm = {}
        start = 0
        for name, arm in self.arms.items():
            end = start + arm.dof
            q_by_arm[name] = q[start:end]
            start = end
        return q_by_arm

    # -----------------------------------------------------------------
    # Mapping interface
    # -----------------------------------------------------------------

    def __getitem__(self, name: str) -> Arm:
        return self.arms[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.arms)

    def __len__(self) -> int:
        return len(self.arms)

    # -----------------------------------------------------------------
    # State queries
    # -----------------------------------------------------------------

    def arm(self, name: str) -> Arm:
        """Get an Arm by name."""
        if name not in self.arms:
            raise ValueError(f"Arm '{name}' not found in this group")
        return self[name]

    @property
    def joint_names(self) -> list[str]:
        """List of joint names for all arms in this group."""
        return [n for arm in self.arms.values() for n in arm.config.joint_names]

    @property
    def extra_arm_body_names(self) -> list[str] | None:
        """List of extra arm body names for collision checking."""
        names = [n for arm in self.arms.values() for n in arm.config.extra_arm_body_names or []]
        return names or None

    @property
    def kinematic_limits(self) -> KinematicLimits:
        """Concatenated kinematic limits for all arms in this group."""
        return KinematicLimits(
            velocity=np.concatenate([arm.config.kinematic_limits.velocity for arm in self.arms.values()]),
            acceleration=np.concatenate([arm.config.kinematic_limits.acceleration for arm in self.arms.values()]),
        )

    @property
    def joint_qpos_indices(self) -> list[int]:
        """List of qpos indices for all arms in this group."""
        return [idx for arm in self.arms.values() for idx in arm.joint_qpos_indices]

    @property
    def ee_site_id_group(self) -> list[int]:
        """List of site IDs for all arms in this group."""
        return [arm.ee_site_id for arm in self.arms.values()]
    
    @property
    def tcp_offset_group(self) -> list[np.ndarray | None]:
        """List of TCP offsets for all arms in this group."""
        return [arm.config.tcp_offset for arm in self.arms.values()]

    @property
    def env(self) -> Environment:
        envs = {arm.env for arm in self.arms.values() if arm.env is not None}
        if not envs:
            raise ValueError("No arms in this group have an associated Environment")
        if len(envs) > 1:
            raise ValueError("Multiple Environments found in this group")
        return next(iter(envs))

    @property
    def grasp_manager(self) -> GraspManager | None:
        """GraspManager shared by all arms in this group, if any."""
        managers = {arm.grasp_manager for arm in self.arms.values() if arm.grasp_manager is not None}
        if not managers:
            return None
        if len(managers) > 1:
            raise ValueError("Multiple GraspManagers found in this group")
        return managers.pop()

    def get_joint_positions(self) -> np.ndarray:
        """Current joint positions (rad)."""
        return np.concatenate([arm.get_joint_positions() for arm in self.arms.values()])

    def set_joint_positions(self, q: np.ndarray, ctx=None) -> None:
        """Set joint positions directly, sync viewer.

        Simulation only — teleports the arm to the target configuration.
        On real hardware, use plan_to_configuration() instead.

        Args:
            q: Joint positions (rad), length must match DOF.
            ctx: ExecutionContext for syncing. If None, runs mj_forward only.
        """
        if len(q) != self.dof:
            raise ValueError(f"Expected {self.dof} joints, got {len(q)}")

        q_by_arm = self._split_by_arm(q)
        for name, arm in self.arms.items():
            arm.set_joint_positions(q_by_arm[name], ctx=ctx)

    def get_joint_velocities(self) -> np.ndarray:
            """Current joint velocities (rad/s)."""
            return np.concatenate([arm.get_joint_velocities() for arm in self.arms.values()])

    def get_ee_pose(self) -> list[np.ndarray]:
        """Current end-effector pose as 4x4 homogeneous transform.

        Calls mj_forward to ensure kinematics are up-to-date, then reads
        the EE site pose. Applies tcp_offset if configured.
        """
        return [arm.get_ee_pose() for arm in self.arms.values()]

    def get_ee_jacobian(self) -> list[np.ndarray]:
        """6xN end-effector Jacobian in world frame.

        Rows 0-2 are linear velocity, rows 3-5 are angular velocity.
        Columns correspond to arm joints in joint_qvel_indices order.
        """
        return [arm.get_ee_jacobian() for arm in self.arms.values()]

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint position limits as (lower, upper) arrays, concatenated
        across all arms in this group (matches self.joint_names order).

        Unlimited joints (limited=False) return [-π, π] as a nominal
        range for sampling/planning. The actual joint has no physical
        limit — angular_joints handles wrapping in the planner.
        """
        los, his = [], []
        for arm in self.arms.values():
            lo, hi = arm.get_joint_limits()
            los.append(lo)
            his.append(hi)
        return np.concatenate(los), np.concatenate(his)

    # -----------------------------------------------------------------
    # Forward kinematics (non-destructive, for planning)
    # -----------------------------------------------------------------

    def forward_kinematics(self, q: np.ndarray) -> list[np.ndarray]:
        """Compute EE pose at configuration q without modifying live state.

        Creates a temporary MjData copy, sets joints to q, runs mj_forward,
        and reads the resulting pose. The live env.data is never touched.
        """
        q_by_arm = self._split_by_arm(q)
        return [arm.forward_kinematics(q_by_arm[name]) for name, arm in self.arms.items()]

    # -----------------------------------------------------------------
    # Planning
    # -----------------------------------------------------------------

    def check_collisions(self, arm_name: str | None = None) -> list[tuple[str, str, float]]:
        """Check current configuration for collisions.

        Uses the group's combined collision checker (cross-arm contacts are
        always detected, regardless of scope), including grasp-aware filtering.

        Args:
            arm_name: If given, only report contacts involving THAT arm's own
                bodies. A contact between this arm and the OTHER arm is still
                included (it does involve this arm's bodies) -- only contacts
                entirely unrelated to this arm (e.g. the other arm hitting the
                table) are excluded. If None (default), every contact in the
                group is reported.

        Returns:
            List of (body, other_body, penetration_mm) tuples, filtered to
            `arm_name` if given. Empty if collision-free. Prints a summary.

        Example::

            robot.both_arms.check_collisions("right")
            # right: 2 contact(s)
            #   forearm_link <-> sugar_box_0: 2.3mm
            #   gripper/pad <-> table: 0.8mm
        """
        if arm_name is not None and arm_name not in self.arms:
            raise ValueError(f"Arm '{arm_name}' not found in this group")

        planner = self.create_planner()
        contacts = planner.collision.get_contacts(self.get_joint_positions())
        label = self.config.name

        if arm_name is not None:
            arm = self.arms[arm_name]
            body_ids = planner.collision.body_ids_for_joints(
                arm.config.joint_names, arm.config.extra_arm_body_names
            )
            body_names = {
                mujoco.mj_id2name(planner.collision.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                for bid in body_ids
            }
            contacts = [(b1, b2, d) for b1, b2, d in contacts if b1 in body_names or b2 in body_names]
            label = arm_name

        if contacts:
            print(f"{label}: {len(contacts)} contact(s)")
            for body, other, depth in contacts:
                body_short = body.split("/", 1)[-1] if "/" in body else body
                other_short = other.split("/", 1)[-1] if "/" in other else other
                print(f"  {body_short} <-> {other_short}: {depth:.1f}mm")
        else:
            print(f"{label}: collision-free")
        return contacts

    def create_planner(
        self,
        config: CBiRRTConfig | None = None,
        *,
        planning_env=None,
    ) -> CBiRRT:
        """Create a thread-safe planner with isolated state.

        Each planner has its own MjData copy and adapters, so multiple
        planners can run in parallel threads.

        Args:
            config: Planner configuration. Defaults built from
                    self.config.planning_defaults.
            planning_env: Pre-forked environment to plan in. When provided,
                used directly instead of forking from self.env. This allows
                the caller to set up custom state (e.g. a different base
                height) without mutating live state.

        Returns:
            Configured CBiRRT planner ready to call .plan().
        """
        if config is None:
            defaults = self.config.planning_defaults
            config = CBiRRTConfig(
                timeout=defaults.timeout,
                max_iterations=defaults.max_iterations,
                step_size=defaults.step_size,
                goal_bias=defaults.goal_bias,
                smoothing_iterations=defaults.smoothing_iterations,
                angular_joints=self._detect_angular_joints(),
            )

        # Use provided env or fork for isolated planning state
        if planning_env is None:
            planning_env = self.env.fork()
        model = planning_env.model
        data = planning_env.data

        # Build adapters
        robot_model = ContextRobotModel(
            model=model,
            data=data,
            joint_qpos_indices=self.joint_qpos_indices,
            ee_site_id_group=self.ee_site_id_group,
            joint_limits=self.get_joint_limits(),
            tcp_offset_group=self.tcp_offset_group,
        )

        # Collision checker with snapshot of current grasp state.
        # Only include objects grasped by THIS arm — objects held by other
        # arms are static obstacles, not part of this arm's robot model.
        extra_bodies = self.extra_arm_body_names
        if self.grasp_manager is not None:
            grasped_objects = frozenset(
                (obj, arm) for obj, arm in self.grasp_manager.grasped.items()
            )
            attachments = {
                obj: att for obj, att in self.grasp_manager._attachments.items() if obj in dict(grasped_objects)
            }
            collision_checker = CollisionChecker(
                model=model,
                data=data,
                joint_names=self.joint_names,
                grasped_objects=grasped_objects,
                attachments=attachments,
                extra_arm_body_names=extra_bodies,
            )
        else:
            collision_checker = CollisionChecker(
                model=model,
                data=data,
                joint_names=self.joint_names,
                extra_arm_body_names=extra_bodies,
            )

        ik = list(self.arms.values())[0].ik_solver if len(self.arms) >= 1 else None
        return CBiRRT(
            robot=robot_model,
            ik_solver=ik if ik is not None else _NoIKSolver(),
            collision_checker=collision_checker,
            config=config,
        )

    def _make_planner_config(
        self,
        timeout: float | None,
        planner_config: CBiRRTConfig | None,
        abort_fn: Callable[[], bool] | None = None,
    ) -> CBiRRTConfig:
        """Build a planner config from planning_defaults, with optional overrides."""
        defaults = self.config.planning_defaults
        if planner_config is not None:
            overrides = {}
            if timeout is not None:
                overrides["timeout"] = timeout
            if abort_fn is not None:
                overrides["abort_fn"] = abort_fn
            return dataclasses.replace(planner_config, **overrides) if overrides else planner_config
        return CBiRRTConfig(
            timeout=timeout if timeout is not None else defaults.timeout,
            max_iterations=defaults.max_iterations,
            step_size=defaults.step_size,
            goal_bias=defaults.goal_bias,
            smoothing_iterations=defaults.smoothing_iterations,
            angular_joints=self._detect_angular_joints(),
            abort_fn=abort_fn,
        )

    def _detect_angular_joints(self) -> tuple[bool, ...] | None:
        """Detect continuous joints for angular distance wrapping.

        A joint is continuous if unlimited OR range > 2π.
        """
        angular = []
        for jname in self.joint_names:
            jid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                angular.append(False)
            elif not self.env.model.jnt_limited[jid]:
                angular.append(True)
            else:
                rng = self.env.model.jnt_range[jid]
                angular.append((rng[1] - rng[0]) > 2 * np.pi * 1.5)
        return tuple(angular) if any(angular) else None


    def retime(
        self,
        path: list[np.ndarray],
        *,
        control_dt: float | None = None,
    ) -> Trajectory:
        """Time-parameterize a path using TOPP-RA.

        Converts a geometric path (from any plan_to_* method) into a
        time-optimal trajectory respecting the arm's kinematic limits.

        Args:
            path: List of waypoint configurations from a planner.
            control_dt: Control timestep for trajectory sampling (default 125 Hz).

        Returns:
            Time-optimal Trajectory with positions, velocities, and accelerations.
        """
        if control_dt is None:
            control_dt = self.config.execution_defaults.control_dt

        limits = self.kinematic_limits
        return Trajectory.from_path(
            path=path,
            vel_limits=limits.velocity,
            acc_limits=limits.acceleration,
            control_dt=control_dt,
            entity=self.config.name,
            joint_names=self.joint_names,
            retime_max_iters=self.config.execution_defaults.retime_max_iters,
            retime_gridpoints=self.config.execution_defaults.retime_gridpoints,
            retime_accel_tol=self.config.execution_defaults.retime_accel_tol,
            retime_shrink_factor=self.config.execution_defaults.retime_shrink_factor,
        )

    def plan_cartesian_path(
        self,
        waypoints: dict[str, list[np.ndarray]],
        *,
        q_start: np.ndarray | None = None,
        control_dt: float = 0.008,
        max_branch_jump: float | None = None,
        partial_ok: bool = False,
        redundancy_window_rad: float = 0.05,
        redundancy_samples: int = 5,
        timeout: float | None = None,
        planner_config: CBiRRTConfig | None = None,
        abort_fn: Callable[[], bool] | None = None,
    ) -> Trajectory:
        """Plan a joint-space trajectory that follows a per-arm SE(3) waypoint
        sequence, with cross-arm collision checking at every combined step.

        Solves analytical IK per named arm at each waypoint index, selects
        each arm's solution closest in joint space to its own previous
        configuration (greedy nearest branch, same as single-arm), combines
        all arms into one joint vector, and rejects the combined step if any
        two arms collide with each other. Unnamed arms hold at their running
        config. The resulting joint-space path is retimed with TOPP-RA.

        All named arms' waypoint lists must be the same length — this
        matches the retract use case, where every arm steps forward once
        per waypoint index in lockstep. There is currently no support for
        per-arm waypoint counts.

        Args:
            waypoints: Map of arm name -> list of 4x4 SE(3) pose matrices in
                world frame, one list per arm that should move. Arms not
                present in this dict hold at their current configuration.
            q_start: Optional combined joint configuration to start from.
                Defaults to ``self.get_joint_positions()``.
            control_dt: Control timestep for trajectory sampling (seconds).
            max_branch_jump: Optional per-arm, per-step joint-space distance
                threshold (radians). Checked per arm, same as single-arm.
                ``None`` disables the check.
            partial_ok: If ``True``, return a trajectory for the longest
                feasible prefix instead of raising when IK, cross-arm
                collision, or ``max_branch_jump`` fails partway through.
                Still raises if the very first waypoint index has no
                feasible combined step.
            redundancy_window_rad: Per-arm locked-joint drift window for
                redundant (EAIK) arms. Same semantics as single-arm.
            redundancy_samples: Per-arm locked-joint sample count. Same
                semantics as single-arm.
            timeout / planner_config / abort_fn: Forwarded to the group's
                collision-checker construction, same as ``plan_to_poses``.

        Returns:
            Time-parameterized combined-arm ``Trajectory`` ready for
            ``ctx.execute``. Under ``partial_ok``, may cover fewer waypoint
            indices than requested.

        Raises:
            ValueError: Unknown arm name, empty/mismatched waypoint lists,
                no feasible IK/collision-free combined step at index 0 (or
                any index without ``partial_ok``).
            RuntimeError: A named arm has no IK solver.
        """
        extra = set(waypoints) - set(self.arms)
        if extra:
            raise ValueError(f"waypoints has unknown arm(s): {sorted(extra)}")
        if not waypoints:
            raise ValueError("plan_cartesian_path: waypoints must be non-empty")

        lengths = {len(v) for v in waypoints.values()}
        if len(lengths) > 1:
            raise ValueError(f"plan_cartesian_path: all arms' waypoint lists must be the same length; got {lengths}")
        num_steps = next(iter(lengths))
        if num_steps == 0:
            raise ValueError("plan_cartesian_path: waypoints must be non-empty")

        for name in waypoints:
            if self.arms[name].ik_solver is None:
                raise RuntimeError(
                    f"plan_cartesian_path requires an arm with an IK solver: arm '{name}' has none"
                )

        arm_order = list(self.arms.keys())
        q_current = self.get_joint_positions().copy() if q_start is None else np.asarray(q_start, dtype=float).copy()
        q_by_arm = self._split_by_arm(q_current)

        config = self._make_planner_config(timeout, planner_config, abort_fn=abort_fn)
        planner = self.create_planner(config)  # used for its collision checker only

        joint_path: list[np.ndarray] = [q_current]

        for i in range(num_steps):
            next_by_arm: dict[str, np.ndarray] = {}
            infeasible_reason: str | None = None

            for name in arm_order:
                arm = self.arms[name]
                q_cur_arm = q_by_arm[name]

                if name not in waypoints:
                    next_by_arm[name] = q_cur_arm  # unnamed/idle arm holds
                    continue

                pose = np.asarray(waypoints[name][i], dtype=float)
                if pose.shape != (4, 4):
                    raise ValueError(f"plan_cartesian_path: arm '{name}' waypoint {i} has shape {pose.shape}, expected (4, 4)")

                ik_kwargs = {"q_init": q_cur_arm}
                locked_idx = getattr(arm.ik_solver, "fixed_joint_index", None)
                if locked_idx is not None:
                    center = float(q_cur_arm[locked_idx])
                    ik_kwargs["discretizations"] = [
                        np.linspace(center - redundancy_window_rad, center + redundancy_window_rad, redundancy_samples)
                    ]

                solutions = arm.ik_solver.solve_valid(pose, **ik_kwargs)
                if not solutions:
                    infeasible_reason = f"no valid IK solution for arm '{name}' at waypoint {i}"
                    break

                q_next_arm = min(solutions, key=lambda q: float(np.linalg.norm(q - q_cur_arm)))

                if max_branch_jump is not None:
                    jump = float(np.linalg.norm(q_next_arm - q_cur_arm))
                    if jump > max_branch_jump:
                        infeasible_reason = (
                            f"IK branch jump for arm '{name}' at waypoint {i} "
                            f"({jump:.3f} rad > max_branch_jump={max_branch_jump:.3f})"
                        )
                        break

                next_by_arm[name] = q_next_arm

            if infeasible_reason is None:
                q_next_combined = np.concatenate([next_by_arm[n] for n in arm_order])
                if not planner.collision.is_valid(q_next_combined):
                    infeasible_reason = f"cross-arm collision at waypoint {i}"

            if infeasible_reason is not None:
                if partial_ok and i > 0:
                    logger.info(
                        "plan_cartesian_path: %s; returning partial trajectory for %d feasible waypoint(s)",
                        infeasible_reason, i,
                    )
                    break
                raise ValueError(f"plan_cartesian_path: {infeasible_reason}")

            joint_path.append(q_next_combined)
            q_by_arm = next_by_arm

        return self.retime(joint_path, control_dt=control_dt)

    # =============================================================================
    # Shared trajectory-planning machinery
    # =============================================================================

    @staticmethod
    def _as_frames(targets: dict) -> list[dict]:
        """Normalize {arm: target | [targets...]} into an ordered list of
        per-frame dicts [{arm: target}, ...].

        A bare target is a length-1 trajectory. Lists are per-arm waypoint
        sequences: all list-valued arms must share one length N; length-1
        (or bare) arms are broadcast (held at that target for every frame).
        Unlike the single-arm plan_to_tsrs, a list here means a SEQUENCE to
        follow, NOT a union of candidates for one waypoint.
        """
        if not targets:
            return []
        norm = {
            name: (list(v) if isinstance(v, (list, tuple)) else [v])
            for name, v in targets.items()
        }
        lengths = {len(v) for v in norm.values()}
        n = max(lengths)
        bad = lengths - {1, n}
        if bad:
            detail = ", ".join(f"{k}: {len(v)}" for k, v in norm.items())
            raise ValueError(
                f"per-arm trajectory lengths must all be 1 (broadcast) or {n}; got {{{detail}}}"
            )
        return [
            {name: (v[i] if len(v) > 1 else v[0]) for name, v in norm.items()}
            for i in range(n)
        ]

    def _combined_goals_for_frame(
        self,
        frame: dict,
        q_start: np.ndarray,
        resolve_candidates: Callable,   # (arm, target, q_ref) -> list[np.ndarray]
        planner: CBiRRT,
    ) -> list[np.ndarray] | None:
        """Resolve one frame into collision-valid COMBINED goal configs.

        Each named arm's target is turned into per-arm config candidates;
        unnamed arms hold at their running config. The cross-product is
        filtered to combinations that are collision-free TOGETHER (this is
        where cross-arm collisions at the goal get rejected).
        """
        extra = set(frame) - set(self.arms)
        if extra:
            raise ValueError(f"goal has unknown arm(s): {sorted(extra)}")

        start_by_arm = self._split_by_arm(q_start)
        arm_order = list(self.arms.keys())

        per_arm: dict[str, list[np.ndarray]] = {}
        for name in arm_order:
            arm = self.arms[name]
            q_ref = start_by_arm[name]
            if name in frame:
                cands = resolve_candidates(arm, frame[name], q_ref)
                if not cands:
                    logger.info("no goal candidates for arm '%s'", name)
                    return None
                per_arm[name] = cands
            else:
                per_arm[name] = [q_ref]  # unnamed arm holds at running config

        combined: list[np.ndarray] = []
        for combo in itertools.product(*(per_arm[n] for n in arm_order)):
            q = np.concatenate(combo)
            if planner.collision.is_valid(q):
                combined.append(q)
                if len(combined) >= self.config.max_bimanual_IK_solutions:
                    break
        return combined or None

    def _plan_frame_sequence(
        self,
        targets: dict,
        resolve_candidates: Callable,
        *,
        constraint_tsrs: list[TSR] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
        abort_fn: Callable[[], bool] | None = None,
        partial_ok: bool = False,
    ) -> list[np.ndarray] | None:
        """Plan a jointly-collision-checked path through a per-arm frame
        sequence. Frames are reached in order; each segment starts where the
        previous one ended. Returns one combined (all-arm) waypoint path.
        """
        if constraint_tsrs is not None:
            raise NotImplementedError("path constraints not supported yet")

        frames = self._as_frames(targets)
        if not frames:
            return None

        config = self._make_planner_config(timeout, planner_config, abort_fn=abort_fn)
        planner = self.create_planner(config)  # one planner, reused across frames

        q_start = self.get_joint_positions()
        full_path: list[np.ndarray] = [q_start]
        for i, frame in enumerate(frames):
            goals = self._combined_goals_for_frame(frame, q_start, resolve_candidates, planner)
            if not goals:
                if partial_ok and i > 0:
                    break
                logger.info("plan frame %d: no collision-free combined goal", i)
                return None
            try:
                segment = planner.plan(start=q_start, goal=goals, seed=seed)
            except (PlanningError, ValueError) as e:
                logger.info("plan frame %d failed: %s", i, e)
                return None
            if segment is None:
                logger.info("plan frame %d failed: no path", i)
                return None

            full_path.extend(segment[1:])  # drop duplicated boundary waypoint
            q_start = full_path[-1]

        logger.info("planned %d-frame trajectory: %d waypoints", len(frames), len(full_path))
        return full_path

    # ---- per-type candidate resolvers ----

    @staticmethod
    def _config_candidates(arm, q_goal, q_ref) -> list[np.ndarray]:
        q = np.asarray(q_goal, dtype=float)
        if q.shape != (arm.dof,):
            raise ValueError(f"arm '{arm.config.name}' expects {arm.dof} joints, got {tuple(q.shape)}")
        return [q]

    def _pose_candidates(self, arm, pose, q_ref) -> list[np.ndarray]:
        if arm.ik_solver is None:
            raise RuntimeError(f"pose planning requires an IK solver on arm '{arm.config.name}'")
        sols = arm.ik_solver.solve_valid(pose) 
        if not sols:
            return []
        sols = sorted(sols, key=lambda q: float(np.linalg.norm(q - q_ref)))
        return sols[: arm.config.max_ik_solutions]

    def _tsr_candidates(self, arm, tsr, q_ref, *, samples: int) -> list[np.ndarray]:
        if arm.ik_solver is None:
            raise RuntimeError(f"TSR planning requires an IK solver on arm '{arm.config.name}'")
        sols: list[np.ndarray] = []
        for _ in range(samples):
            sols.extend(arm.ik_solver.solve_valid(tsr.sample()))  # tsr.sample() -> 4x4 pose
        if not sols:
            return []
        sols = sorted(sols, key=lambda q: float(np.linalg.norm(q - q_ref)))
        return sols[: arm.config.max_ik_solutions]

    # =============================================================================
    # Public planning API — each accepts a single target or a trajectory per arm
    # =============================================================================

    def plan_to_configuration(
        self,
        goal: dict[str, np.ndarray | list[np.ndarray]],
        *,
        constraint_tsrs: list[TSR] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
        abort_fn: Callable[[], bool] | None = None,
    ) -> list[np.ndarray] | None:
        """Plan to per-arm joint configuration(s).

        goal maps arm name -> a joint config, or a list of joint configs to
        pass through in order. Unnamed arms hold at their current config.
        """
        return self._plan_frame_sequence(
            goal, self._config_candidates,
            constraint_tsrs=constraint_tsrs, timeout=timeout,
            seed=seed, planner_config=planner_config, abort_fn=abort_fn,
        )

    def plan_to_poses(
        self,
        goal: dict[str, np.ndarray | list[np.ndarray]],
        *,
        constraint_tsrs: list[TSR] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
        abort_fn: Callable[[], bool] | None = None,
        partial_ok: bool = False,
    ) -> list[np.ndarray] | None:
        """Plan to per-arm end-effector pose(s).

        goal maps arm name -> a 4x4 pose, or a list of 4x4 poses to pass
        through in order. IK is solved per arm and combinations are checked
        for cross-arm collision at each frame. Unnamed arms hold.
        """
        return self._plan_frame_sequence(
            goal, self._pose_candidates,
            constraint_tsrs=constraint_tsrs, timeout=timeout,
            seed=seed, planner_config=planner_config, abort_fn=abort_fn,
            partial_ok=partial_ok,
        )

    def plan_to_tsrs(
        self,
        goal: dict[str, TSR | list[TSR]],
        *,
        samples: int = 8,
        constraint_tsrs: list[TSR] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
        abort_fn: Callable[[], bool] | None = None,
    ) -> list[np.ndarray] | None:
        """Plan to per-arm TSR goal region(s).

        goal maps arm name -> a TSR, or a list of TSRs to pass through in
        order. Each TSR is sampled `samples` times and IK-solved to build
        per-arm candidates. Unnamed arms hold. Note: a list is a SEQUENCE of
        waypoints, not a union of candidates (unlike Arm.plan_to_tsrs).
        """
        resolver = lambda arm, tsr, q_ref: self._tsr_candidates(arm, tsr, q_ref, samples=samples)
        return self._plan_frame_sequence(
            goal, resolver,
            constraint_tsrs=constraint_tsrs, timeout=timeout,
            seed=seed, planner_config=planner_config, abort_fn=abort_fn,
        )

class _NoIKSolver:
    """Stub IK solver that returns no solutions.

    Used when no IK solver is injected. Config-to-config planning
    still works; only pose/TSR-based planning requires real IK.
    """

    def solve(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        return []

    def solve_valid(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        return self.solve(pose, q_init)
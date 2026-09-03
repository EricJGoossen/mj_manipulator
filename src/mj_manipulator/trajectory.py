# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Trajectory representation and time-optimal retiming using TOPP-RA."""

from dataclasses import dataclass

import numpy as np
import toppra
import toppra.algorithm as algo
import toppra.constraint as constraint


@dataclass
class Trajectory:
    """Time-parameterized robot trajectory.

    Stores positions, velocities, accelerations at discrete timestamps.
    Compatible with both MuJoCo simulation and real robot execution.

    The trajectory is dense (waypoints at control frequency) to avoid
    needing spline evaluation in the control loop.

    Entity information (entity name and joint_names) enables hardware
    deployment by identifying which joints to command.
    """

    timestamps: np.ndarray  # (N,) seconds from start
    positions: np.ndarray  # (N, dof) joint positions in radians
    velocities: np.ndarray  # (N, dof) joint velocities in rad/s
    accelerations: np.ndarray  # (N, dof) joint accelerations in rad/s²
    entity: str | None = None  # Entity name: "left_arm", "right_base", etc.
    joint_names: list[str] | None = None  # MuJoCo joint names for validation

    def __post_init__(self):
        """Validate trajectory dimensions."""
        n_waypoints = len(self.timestamps)
        if self.positions.shape[0] != n_waypoints:
            raise ValueError(f"Position shape {self.positions.shape} doesn't match timestamps length {n_waypoints}")
        if self.velocities.shape[0] != n_waypoints:
            raise ValueError(f"Velocity shape {self.velocities.shape} doesn't match timestamps length {n_waypoints}")
        if self.accelerations.shape[0] != n_waypoints:
            raise ValueError(
                f"Acceleration shape {self.accelerations.shape} doesn't match timestamps length {n_waypoints}"
            )

        if self.positions.shape[1] != self.velocities.shape[1]:
            raise ValueError(
                f"DOF mismatch: positions {self.positions.shape[1]} vs velocities {self.velocities.shape[1]}"
            )

        # Validate joint_names length matches DOF if provided
        if self.joint_names is not None:
            if len(self.joint_names) != self.positions.shape[1]:
                raise ValueError(
                    f"joint_names length {len(self.joint_names)} doesn't match DOF {self.positions.shape[1]}"
                )

    @property
    def duration(self) -> float:
        """Total duration of trajectory in seconds."""
        return float(self.timestamps[-1])

    @property
    def dof(self) -> int:
        """Degrees of freedom (number of joints)."""
        return self.positions.shape[1]

    @property
    def num_waypoints(self) -> int:
        """Number of waypoints in trajectory."""
        return len(self.timestamps)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate position, velocity, acceleration at time t.

        Args:
            t: Time in seconds (clamped to [0, duration])

        Returns:
            Tuple of (position, velocity, acceleration) arrays
        """
        t = np.clip(t, 0.0, self.duration)

        idx = np.searchsorted(self.timestamps, t)

        if idx == 0:
            return self.positions[0], self.velocities[0], self.accelerations[0]
        if idx >= len(self.timestamps):
            return self.positions[-1], self.velocities[-1], self.accelerations[-1]

        t0, t1 = self.timestamps[idx - 1], self.timestamps[idx]
        alpha = (t - t0) / (t1 - t0)

        pos = (1 - alpha) * self.positions[idx - 1] + alpha * self.positions[idx]
        vel = (1 - alpha) * self.velocities[idx - 1] + alpha * self.velocities[idx]
        acc = (1 - alpha) * self.accelerations[idx - 1] + alpha * self.accelerations[idx]

        return pos, vel, acc

    def split_trajectory(self, arm_group) -> dict[str, "Trajectory"]:
        """Split a trajectory into per-arm trajectories.

        Args:
            arm_group: The ArmGroup containing the arms to split for.

        Returns:
            A dictionary mapping arm names to their corresponding Trajectory.
        """
        if self.positions.shape[1] != sum(arm.dof for arm in arm_group.arms.values()):
            raise ValueError(
                f"Trajectory DOF {self.positions.shape[1]} doesn't match total DOF of arm group {sum(arm.dof for arm in arm_group.arms.values())}"
            )

        per_arm_trajectories = {}
        total_dof = 0
        for arm_name in arm_group.keys():
            arm_dof = arm_group[arm_name].dof
            total_dof += arm_dof

            # Check all names in trajectory_names are in arm_group.arms[arm_name].joint_names
            trajectory_names = None
            if self.joint_names is not None:
                trajectory_names = self.joint_names[total_dof - arm_dof:total_dof]

                if trajectory_names != list(arm_group[arm_name].config.joint_names):
                    raise ValueError(
                        f"Joint names for arm '{arm_name}' don't match: "
                        f"trajectory slice has {trajectory_names}, arm expects "
                        f"{list(arm_group[arm_name].config.joint_names)}"
                    )

            per_arm_trajectories[arm_name] = Trajectory(
                timestamps=self.timestamps,
                positions=self.positions[:, total_dof - arm_dof:total_dof],
                velocities=self.velocities[:, total_dof - arm_dof:total_dof],
                accelerations=self.accelerations[:, total_dof - arm_dof:total_dof],
                entity=arm_name,
                joint_names=trajectory_names,
            )
        return per_arm_trajectories

    @classmethod
    def from_path(
        cls,
        path: list[np.ndarray],
        vel_limits: np.ndarray,
        acc_limits: np.ndarray,
        control_dt: float = 0.008,
        entity: str | None = None,
        joint_names: list[str] | None = None,
        retime_max_iters: int = 8,
        retime_gridpoints: int = 1000,
        retime_accel_tol: float = 1e-3,
        retime_shrink_factor: float = 0.97,
        collision_checker=None,
        collision_max_densify: int = 4,
    ) -> "Trajectory":
        """Create time-optimal trajectory from geometric path using TOPP-RA.

        The input `path` is only guaranteed collision-free at the straight-line
        edges the planner checked -- retiming fits a smooth cubic spline
        (`toppra.SplineInterpolator`) through those waypoints, and that spline
        can bow outward between waypoints (e.g. around a corner near an
        obstacle) into space the planner never validated. If `collision_checker`
        is given, every sampled position of the retimed trajectory is checked
        against it; on a collision, the path is densified (a midpoint is
        inserted between every consecutive pair of waypoints, halving the
        spline's freedom to deviate from the checked straight lines) and
        retiming is retried, up to `collision_max_densify` times.

        Args:
            path: List of waypoint configurations (joint angles in radians)
            vel_limits: Joint velocity limits in rad/s (shape: (dof,))
            acc_limits: Joint acceleration limits in rad/s² (shape: (dof,))
            control_dt: Control timestep in seconds (default: 125 Hz)
            entity: Entity name for hardware deployment
            joint_names: MuJoCo joint names for validation
            collision_checker: Optional object with `.get_contacts(q)` (the
                mj_manipulator CollisionChecker protocol). When given, the
                retimed trajectory is validated and re-densified on collision
                instead of being returned as-is.
            collision_max_densify: Maximum number of densify-and-retry rounds
                when `collision_checker` is given.

        Returns:
            Trajectory with time-optimal parameterization respecting limits

        Raises:
            ValueError: If path is empty or has inconsistent dimensions
            RuntimeError: If TOPP-RA fails to find a valid parameterization, or
                if `collision_checker` still reports a collision after
                `collision_max_densify` densify rounds
        """
        if not path:
            raise ValueError("Path cannot be empty")

        path_array = np.array(path)
        if path_array.ndim != 2:
            raise ValueError(f"Path must be 2D array, got shape {path_array.shape}")

        dof = path_array.shape[1]
        if len(vel_limits) != dof:
            raise ValueError(f"Velocity limits dimension {len(vel_limits)} doesn't match path DOF {dof}")
        if len(acc_limits) != dof:
            raise ValueError(f"Acceleration limits dimension {len(acc_limits)} doesn't match path DOF {dof}")

        # Remove consecutive duplicate waypoints
        filtered_path = [path_array[0]]
        for i in range(1, len(path_array)):
            if not np.allclose(path_array[i], filtered_path[-1], atol=1e-10):
                filtered_path.append(path_array[i])

        path_array = np.array(filtered_path)

        # Trivial trajectory if already at goal
        if len(path_array) < 2:
            return cls(
                timestamps=np.array([0.0]),
                positions=path_array,
                velocities=np.zeros_like(path_array),
                accelerations=np.zeros_like(path_array),
                entity=entity,
                joint_names=joint_names,
            )

        for densify_attempt in range(collision_max_densify + 1):
            path_positions = toppra.SplineInterpolator(np.linspace(0, 1, len(path_array)), path_array)

            # The solve grid must stay dense relative to the knot count (see
            # _solve_toppra's docstring) -- a fixed retime_gridpoints becomes
            # too coarse once densify has multiplied the knot count, and
            # produces spurious, unconverging acceleration violations below.
            n_gridpoints = max(retime_gridpoints, 10 * len(path_array))

            working_acc_limits = np.asarray(acc_limits, dtype=float).copy()

            for attempt in range(retime_max_iters):
                timestamps, positions, velocities, accelerations = cls._solve_toppra(
                    path_positions, vel_limits, working_acc_limits, control_dt,
                    n_gridpoints=n_gridpoints,
                )
                amax = np.max(np.abs(accelerations), axis=0)
                violated = amax > acc_limits * (1.0 + retime_accel_tol)
                if not violated.any():
                    break

                working_acc_limits[violated] *= retime_shrink_factor
            else:
                raise RuntimeError(
                    f"TOPP-RA could not produce a trajectory within acceleration limits after "
                    f"{retime_max_iters} retighten attempts (joint(s) "
                    f"{np.where(violated)[0].tolist()} still over limit). This means the path "
                    "itself likely demands more acceleration than the joint is rated for, not "
                    "just a reparametrization artifact -- check the path, not just the retimer."
                )

            if collision_checker is None:
                break

            collision = cls._first_collision(collision_checker, positions)
            if collision is None:
                break

            if densify_attempt == collision_max_densify:
                idx, contacts = collision
                raise RuntimeError(
                    f"TOPP-RA retiming produced a trajectory that collides at waypoint "
                    f"{idx}/{len(positions)} ({contacts}) even after densifying the input "
                    f"path from {len(path)} to {len(path_array)} waypoints. The geometric "
                    "path itself was collision-free at the checked resolution, so the spline "
                    "fit through it is bowing into the obstacle between waypoints -- consider "
                    "a smaller planner step_size or a collision safety margin."
                )

            path_array = cls._densify_path(path_array)

        return cls(
            timestamps=timestamps,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            entity=entity,
            joint_names=joint_names,
        )

    @staticmethod
    def _densify_path(path_array: np.ndarray) -> np.ndarray:
        """Insert a midpoint between every consecutive waypoint pair.

        Halves the spacing the spline fit has to bridge, which shrinks how
        far a cubic spline through the path can bow away from the original
        (collision-checked) straight-line segments.
        """
        midpoints = (path_array[:-1] + path_array[1:]) / 2.0
        densified = np.empty((2 * len(path_array) - 1, path_array.shape[1]), dtype=path_array.dtype)
        densified[0::2] = path_array
        densified[1::2] = midpoints
        return densified

    @staticmethod
    def _first_collision(collision_checker, positions: np.ndarray):
        """Return (index, contacts) for the first colliding sample, or None."""
        for i, q in enumerate(positions):
            contacts = collision_checker.get_contacts(q)
            if contacts:
                return i, contacts
        return None

    @staticmethod
    def _solve_toppra(path_positions, vel_limits, acc_limits, control_dt, n_gridpoints):
        """Run TOPP-RA once at a given per-joint acceleration bound and an
        explicit, dense solve grid.

        The dense grid isn't just about accuracy: leaving `gridpoints`
        unset lets TOPP-RA derive its solve grid from the path's own knot
        spacing, which for a sparse/uneven CBiRRT path can be coarse enough
        to produce a genuinely slower-than-optimal parametrization on top
        of the sampling inaccuracy -- an explicit dense grid fixes both at
        once (see the empirical comparison this replaced: coarse-grid
        durations were 30-50% longer than dense-grid ones, independent of
        the accel-limit issue).
        """
        vel_limits_minmax = np.stack((-vel_limits, vel_limits)).T
        acc_limits_minmax = np.stack((-acc_limits, acc_limits)).T

        pc_vel = constraint.JointVelocityConstraint(vel_limits_minmax)
        pc_acc = constraint.JointAccelerationConstraint(acc_limits_minmax)

        instance = algo.TOPPRA(
            [pc_vel, pc_acc],
            path_positions,
            gridpoints=np.linspace(0, 1, n_gridpoints),
            parametrizer="ParametrizeConstAccel",
        )

        jnt_traj = instance.compute_trajectory()
        if jnt_traj is None:
            raise RuntimeError(
                "TOPP-RA failed to find valid trajectory. Path may violate velocity or acceleration constraints."
            )

        duration = jnt_traj.duration
        timestamps = np.arange(0.0, duration, control_dt)
        if not np.isclose(timestamps[-1], duration, rtol=0.0, atol=1e-8):
            timestamps = np.append(timestamps, duration)

        return timestamps, jnt_traj(timestamps), jnt_traj(timestamps, 1), jnt_traj(timestamps, 2)

def create_linear_trajectory(
    start: float,
    end: float,
    vel_limit: float,
    acc_limit: float,
    control_dt: float = 0.008,
    entity: str | None = None,
    joint_names: list[str] | None = None,
) -> Trajectory:
    """Generate trapezoidal velocity profile for 1D linear motion.

    Three phases: acceleration, cruise (if distance allows), deceleration.
    For short distances, becomes triangular (no cruise phase).

    Args:
        start: Starting position in meters
        end: Target position in meters
        vel_limit: Maximum velocity in m/s
        acc_limit: Maximum acceleration in m/s²
        control_dt: Control timestep in seconds (default: 125 Hz)
        entity: Entity name for hardware deployment
        joint_names: MuJoCo joint names for validation

    Returns:
        Trajectory object with 1 DOF
    """
    distance = abs(end - start)
    direction = 1.0 if end > start else -1.0

    if distance < 1e-8:
        return Trajectory(
            timestamps=np.array([0.0]),
            positions=np.array([[start]]),
            velocities=np.array([[0.0]]),
            accelerations=np.array([[0.0]]),
            entity=entity,
            joint_names=joint_names,
        )

    t_accel = vel_limit / acc_limit
    d_accel = 0.5 * acc_limit * t_accel**2

    if 2 * d_accel <= distance:
        # Trapezoidal profile
        d_cruise = distance - 2 * d_accel
        t_cruise = d_cruise / vel_limit
        t_total = 2 * t_accel + t_cruise
    else:
        # Triangular profile
        t_accel = np.sqrt(distance / acc_limit)
        vel_limit = acc_limit * t_accel
        d_accel = 0.5 * distance
        t_cruise = 0.0
        t_total = 2 * t_accel

    timestamps = np.arange(0.0, t_total, control_dt)
    if not np.isclose(timestamps[-1], t_total, atol=1e-8):
        timestamps = np.append(timestamps, t_total)

    positions = []
    velocities = []
    accelerations = []

    for t in timestamps:
        if t <= t_accel:
            p = start + direction * 0.5 * acc_limit * t**2
            v = direction * acc_limit * t
            a = direction * acc_limit
        elif t <= t_accel + t_cruise:
            t_in_cruise = t - t_accel
            p = start + direction * (d_accel + vel_limit * t_in_cruise)
            v = direction * vel_limit
            a = 0.0
        else:
            t_remaining = t_total - t
            p = end - direction * 0.5 * acc_limit * t_remaining**2
            v = direction * acc_limit * t_remaining
            a = -direction * acc_limit

        positions.append([p])
        velocities.append([v])
        accelerations.append([a])

    return Trajectory(
        timestamps=timestamps,
        positions=np.array(positions),
        velocities=np.array(velocities),
        accelerations=np.array(accelerations),
        entity=entity,
        joint_names=joint_names,
    )

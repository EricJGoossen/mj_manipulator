# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for SimContext's concurrent multi-arm trajectory execution.

Covers the PlanGroupResult path added to SimContext.execute():

- Tick-driven mode (event loop + controller): each arm's trajectory runs
  as its own non-blocking runner, all started together and awaited
  together. A _SiblingFailureSignal ties them so one arm's abort/failure
  stops every other arm in the same batch. A pass-1 ownership check
  refuses to start ANY trajectory if ANY entity is unavailable — all or
  nothing, not "start what we can".
- Legacy mode (no event loop): a single-threaded loop samples every arm's
  trajectory at the reference (first-arm) timestamps each iteration, so
  motion is visually simultaneous even though nothing is threaded. Unlike
  the tick-driven path, per-arm timestamps don't need to match exactly —
  each trajectory is independently interpolated via Trajectory.sample().

Uses two single-joint "arms" carved out of the shared 2-joint conftest
model (joint1/act1 -> "left", joint2/act2 -> "right") so the two arms
are physically independent and their final positions can be checked
without cross-talk, following the two-MockArm pattern already used in
test_event_loop.py / test_kinematic_controller.py.
"""

from __future__ import annotations

import threading
import time

import mujoco
import numpy as np
import pytest
from conftest import MockConfig

from mj_manipulator.config import PhysicsConfig, PhysicsExecutionConfig
from mj_manipulator.event_loop import PhysicsEventLoop
from mj_manipulator.ownership import OwnerKind
from mj_manipulator.planning import PlanGroupResult
from mj_manipulator.sim_context import SimContext
from mj_manipulator.trajectory import Trajectory


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class SingleJointArm:
    """Minimal single-joint arm-like object, carved out of the shared
    2-joint conftest model. Two of these (joint1/act1, joint2/act2) model
    two physically-independent arms sharing one MuJoCo model, so concurrent
    execution can be verified by checking each arm ends at its own target
    without disturbing the other.
    """

    def __init__(self, name, model, data, joint_name, actuator_name):
        self.config = MockConfig(name=name)

        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        self.joint_qpos_indices = [model.jnt_qposadr[j]]
        self.joint_qvel_indices = [model.jnt_dofadr[j]]
        self.actuator_ids = [a]

        self.dof = 1
        self.gripper = None
        self.grasp_manager = None
        self._data = data

    def get_joint_positions(self):
        return np.array([self._data.qpos[idx] for idx in self.joint_qpos_indices])


@pytest.fixture
def left_arm(model_and_data):
    model, data = model_and_data
    return SingleJointArm("left", model, data, "joint1", "act1")


@pytest.fixture
def right_arm(model_and_data):
    model, data = model_and_data
    return SingleJointArm("right", model, data, "joint2", "act2")


def _traj(start: float, end: float, *, entity: str, joint_name: str, n: int = 5) -> Trajectory:
    """Build a single-DOF Trajectory from start to end over n waypoints."""
    positions = np.linspace(start, end, n).reshape(-1, 1)
    return Trajectory(
        timestamps=np.linspace(0.0, 1.0, n),
        positions=positions,
        velocities=np.zeros_like(positions),
        accelerations=np.zeros_like(positions),
        joint_names=[joint_name],
        entity=entity,
    )


def _group(left: Trajectory, right: Trajectory) -> PlanGroupResult:
    return PlanGroupResult.from_trajectories({"left": left, "right": right})


# ---------------------------------------------------------------------------
# Tick-driven mode (event loop present)
# ---------------------------------------------------------------------------


class TestExecutePlanGroupResultTickDrivenKinematic:
    """Kinematic controller + event loop -- exact tracking, deterministic."""

    def test_both_arms_reach_their_targets(self, model_and_data, left_arm, right_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"left": left_arm, "right": right_arm},
            physics=False,
            headless=True,
            event_loop=PhysicsEventLoop(),
        ) as ctx:
            group = _group(
                _traj(0.0, 0.3, entity="left", joint_name="joint1"),
                _traj(0.0, -0.2, entity="right", joint_name="joint2"),
            )
            assert ctx.execute(group) is True
            assert abs(data.qpos[left_arm.joint_qpos_indices[0]] - 0.3) < 1e-6
            assert abs(data.qpos[right_arm.joint_qpos_indices[0]] - (-0.2)) < 1e-6

    def test_mismatched_timestamps_raises(self, model_and_data, left_arm, right_arm):
        """The tick-driven path requires every trajectory in the batch to
        share identical timestamps -- unlike the legacy no-event-loop path
        (see TestExecutePlanGroupResultLegacyPath), it does not interpolate
        per-arm, so it refuses batches that don't line up."""
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"left": left_arm, "right": right_arm},
            physics=False,
            headless=True,
            event_loop=PhysicsEventLoop(),
        ) as ctx:
            left = Trajectory(
                timestamps=np.array([0.0, 0.5, 1.0]),
                positions=np.array([[0.0], [0.1], [0.2]]),
                velocities=np.zeros((3, 1)),
                accelerations=np.zeros((3, 1)),
                joint_names=["joint1"],
                entity="left",
            )
            right = Trajectory(
                timestamps=np.array([0.0, 0.3, 1.0]),  # differs at index 1
                positions=np.array([[0.0], [-0.1], [-0.2]]),
                velocities=np.zeros((3, 1)),
                accelerations=np.zeros((3, 1)),
                joint_names=["joint2"],
                entity="right",
            )
            with pytest.raises(ValueError, match="timestamps"):
                ctx.execute(_group(left, right))

    def test_ownership_released_after_success(self, model_and_data, left_arm, right_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"left": left_arm, "right": right_arm},
            physics=False,
            headless=True,
            event_loop=PhysicsEventLoop(),
        ) as ctx:
            group = _group(
                _traj(0.0, 0.1, entity="left", joint_name="joint1"),
                _traj(0.0, -0.1, entity="right", joint_name="joint2"),
            )
            assert ctx.execute(group) is True
            assert ctx.ownership.owner_of("left")[0] == OwnerKind.IDLE
            assert ctx.ownership.owner_of("right")[0] == OwnerKind.IDLE

    def test_refuses_all_if_any_entity_already_owned(self, model_and_data, left_arm, right_arm):
        """All-or-nothing pre-flight: if ANY arm in the batch is unavailable,
        NO arm in the batch starts -- "right" must stay exactly as it was
        (still owned by the other operation) and "left" must never be
        acquired at all."""
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"left": left_arm, "right": right_arm},
            physics=False,
            headless=True,
            event_loop=PhysicsEventLoop(),
        ) as ctx:
            other_owner = object()
            ctx.ownership.acquire("right", OwnerKind.TRAJECTORY, other_owner)

            group = _group(
                _traj(0.0, 0.3, entity="left", joint_name="joint1"),
                _traj(0.0, -0.3, entity="right", joint_name="joint2"),
            )
            assert ctx.execute(group) is False

            # "left" was never touched.
            assert ctx.ownership.owner_of("left")[0] == OwnerKind.IDLE
            assert abs(data.qpos[left_arm.joint_qpos_indices[0]]) < 1e-9
            # "right" is exactly as it was before the call.
            kind, owner = ctx.ownership.owner_of("right")
            assert kind == OwnerKind.TRAJECTORY
            assert owner is other_owner

    def test_caller_abort_fn_aborts_all_running_siblings(self, model_and_data, left_arm, right_arm):
        """A single caller-supplied abort_fn composes into every arm's
        per-tick abort predicate, so flipping it once stops every arm
        running in the batch, not just one."""
        model, data = model_and_data
        event_loop = PhysicsEventLoop()
        with SimContext(
            model,
            data,
            {"left": left_arm, "right": right_arm},
            physics=False,
            headless=True,
            event_loop=event_loop,
        ) as ctx:
            group = _group(
                _traj(0.0, 0.4, entity="left", joint_name="joint1", n=10),
                _traj(0.0, -0.4, entity="right", joint_name="joint2", n=10),
            )
            result_holder: dict[str, bool] = {}
            stop = threading.Event()

            def run_execute() -> None:
                result_holder["result"] = ctx.execute(group, abort_fn=stop.is_set)

            t = threading.Thread(target=run_execute, daemon=True)
            t.start()
            try:
                for _ in range(2):
                    event_loop.tick()
                stop.set()

                deadline = time.monotonic() + 5.0
                while t.is_alive() and time.monotonic() < deadline:
                    event_loop.tick()
            finally:
                t.join(timeout=5.0)

            assert not t.is_alive()
            assert result_holder["result"] is False
            # Neither arm reached its final waypoint.
            assert abs(data.qpos[left_arm.joint_qpos_indices[0]] - 0.4) > 0.05
            assert abs(data.qpos[right_arm.joint_qpos_indices[0]] - (-0.4)) > 0.05
            # Both released back to idle -- an abort_fn abort is not an
            # ownership preemption, so nobody else claimed the arms.
            assert ctx.ownership.owner_of("left")[0] == OwnerKind.IDLE
            assert ctx.ownership.owner_of("right")[0] == OwnerKind.IDLE

    def test_sibling_failure_aborts_still_running_arm(self, model_and_data, left_arm, right_arm):
        """One arm's trajectory is preempted mid-flight (as teleop
        activation would do). The _SiblingFailureSignal must propagate
        that failure to the still-running sibling so it stops early too,
        instead of running to completion on its own."""
        model, data = model_and_data
        event_loop = PhysicsEventLoop()
        with SimContext(
            model,
            data,
            {"left": left_arm, "right": right_arm},
            physics=False,
            headless=True,
            event_loop=event_loop,
        ) as ctx:
            group = _group(
                _traj(0.0, 0.4, entity="left", joint_name="joint1", n=10),
                _traj(0.0, -0.4, entity="right", joint_name="joint2", n=10),
            )
            result_holder: dict[str, bool] = {}

            def run_execute() -> None:
                result_holder["result"] = ctx.execute(group)

            t = threading.Thread(target=run_execute, daemon=True)
            t.start()
            try:
                # Let both runners start and make some (not full) progress.
                for _ in range(2):
                    event_loop.tick()

                # Simulate teleop grabbing "left" mid-trajectory.
                teleop_owner = object()
                ctx.ownership.preempt("left", OwnerKind.TELEOP, teleop_owner)

                deadline = time.monotonic() + 5.0
                while t.is_alive() and time.monotonic() < deadline:
                    event_loop.tick()
            finally:
                t.join(timeout=5.0)

            assert not t.is_alive()
            assert result_holder["result"] is False

            # "right" was aborted by the sibling-failure signal, not by
            # reaching its own final waypoint.
            assert abs(data.qpos[right_arm.joint_qpos_indices[0]] - (-0.4)) > 0.05

            # "right" releases back to idle (still owned by the trajectory
            # when it aborted); "left" stays with teleop -- execute() only
            # releases arms it still owns, and preempt() took "left" away.
            assert ctx.ownership.owner_of("right")[0] == OwnerKind.IDLE
            assert ctx.ownership.owner_of("left")[0] == OwnerKind.TELEOP


class TestExecutePlanGroupResultTickDrivenPhysics:
    """PhysicsController + event loop -- real actuator convergence for
    two independently-actuated joints running in the same batch."""

    def test_both_arms_converge_concurrently(self, model_and_data, left_arm, right_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"left": left_arm, "right": right_arm},
            headless=True,
            physics_config=PhysicsConfig(
                execution=PhysicsExecutionConfig(
                    control_dt=0.002,
                    position_tolerance=0.3,
                    velocity_tolerance=1.0,
                    convergence_timeout_steps=2000,
                ),
            ),
            event_loop=PhysicsEventLoop(),
        ) as ctx:
            group = _group(
                _traj(0.0, 0.3, entity="left", joint_name="joint1"),
                _traj(0.0, -0.3, entity="right", joint_name="joint2"),
            )
            assert ctx.execute(group) is True
            # Position actuators under gravity settle with steady-state P-gain
            # error, so this doesn't check exact convergence (see
            # test_execute_trajectory_physics in test_sim_context.py for the
            # same reasoning) -- just that both arms actually moved, in the
            # right direction, independently and concurrently.
            assert data.qpos[left_arm.joint_qpos_indices[0]] > 0.02
            assert data.qpos[right_arm.joint_qpos_indices[0]] < -0.02


# ---------------------------------------------------------------------------
# Legacy mode (no event loop) -- single-threaded, interleaved sampling
# ---------------------------------------------------------------------------


class TestExecutePlanGroupResultLegacyPath:
    def test_both_arms_reach_their_targets(self, model_and_data, left_arm, right_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"left": left_arm, "right": right_arm},
            physics=False,
            headless=True,
        ) as ctx:
            group = _group(
                _traj(0.0, 0.3, entity="left", joint_name="joint1"),
                _traj(0.0, -0.3, entity="right", joint_name="joint2"),
            )
            assert ctx.execute(group) is True
            assert abs(data.qpos[left_arm.joint_qpos_indices[0]] - 0.3) < 1e-6
            assert abs(data.qpos[right_arm.joint_qpos_indices[0]] - (-0.3)) < 1e-6

    def test_tolerates_per_arm_timestamps_that_dont_match(self, model_and_data, left_arm, right_arm):
        """Unlike the tick-driven path, the no-event-loop legacy path
        samples every arm's own Trajectory (via Trajectory.sample(), which
        interpolates on that trajectory's own timestamps) at a shared
        reference time sequence taken from the first arm. Per-arm timing
        doesn't need to match -- it must NOT raise, and a shorter-timestamp
        arm is only sampled partway through its own motion."""
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"left": left_arm, "right": right_arm},
            physics=False,
            headless=True,
        ) as ctx:
            left = Trajectory(
                timestamps=np.array([0.0, 1.0]),
                positions=np.array([[0.0], [0.4]]),
                velocities=np.zeros((2, 1)),
                accelerations=np.zeros((2, 1)),
                joint_names=["joint1"],
                entity="left",
            )
            right = Trajectory(
                timestamps=np.array([0.0, 2.0]),  # a different duration entirely
                positions=np.array([[0.0], [-0.8]]),
                velocities=np.zeros((2, 1)),
                accelerations=np.zeros((2, 1)),
                joint_names=["joint2"],
                entity="right",
            )
            assert ctx.execute(_group(left, right)) is True  # does not raise

            # left completes fully -- the reference times come from left.
            assert abs(data.qpos[left_arm.joint_qpos_indices[0]] - 0.4) < 1e-6
            # right is only sampled up to t=1.0 of its own 2.0s duration,
            # i.e. halfway -- NOT its final position of -0.8.
            assert abs(data.qpos[right_arm.joint_qpos_indices[0]] - (-0.4)) < 1e-6

    def test_context_abort_fn_short_circuits_before_any_motion(self, model_and_data, left_arm, right_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"left": left_arm, "right": right_arm},
            physics=False,
            headless=True,
            abort_fn=lambda: True,
        ) as ctx:
            group = _group(
                _traj(0.0, 0.3, entity="left", joint_name="joint1"),
                _traj(0.0, -0.3, entity="right", joint_name="joint2"),
            )
            assert ctx.execute(group) is False
            assert abs(data.qpos[left_arm.joint_qpos_indices[0]]) < 1e-9
            assert abs(data.qpos[right_arm.joint_qpos_indices[0]]) < 1e-9

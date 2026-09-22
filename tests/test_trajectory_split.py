"""Tests for Trajectory.split_trajectory() -- the method-on-Trajectory design.

Superseding notes vs. an earlier standalone-function version:
- API changed: this is combined.split_trajectory(arm_group), a method,
  not split_trajectory(combined, arm_group). Every call site differs.
- The joint-name-order test needed a NEW case: swapping order WITHIN one
  arm's own slice, not swapping across arm boundaries. A full-list
  reversal is caught even by a plain membership check (none of the names
  belong to that arm at all), so it never actually exercised the
  positional-equality fix.
"""
from __future__ import annotations

import numpy as np
import pytest

from mj_manipulator.trajectory import Trajectory
from mj_manipulator.arm_group import ArmGroup
from mj_manipulator.config import ArmConfig, ArmGroupConfig, KinematicLimits, PlanningDefaults


class FakeArm:
    def __init__(self, name: str, dof: int):
        self.dof = dof
        self.config = ArmConfig(
            name=name, entity_type="arm",
            joint_names=[f"{name}_j{i}" for i in range(dof)],
            kinematic_limits=KinematicLimits(velocity=np.full(dof, 1.0), acceleration=np.full(dof, 2.0)),
        )
        self.env = None
        self.grasp_manager = None


def make_group(left_dof: int, right_dof: int) -> ArmGroup:
    left = FakeArm("left", left_dof)
    right = FakeArm("right", right_dof)
    config = ArmGroupConfig(
        name="bimanual", entity_type="arm_group",
        joint_names=left.config.joint_names + right.config.joint_names,
        planning_defaults=PlanningDefaults(),
    )
    return ArmGroup({"left": left, "right": right}, config)


def make_combined(group: ArmGroup, n_waypoints: int = 5) -> Trajectory:
    dof = group.dof
    timestamps = np.linspace(0.0, 1.0, n_waypoints)
    base = np.tile(np.arange(dof, dtype=float), (n_waypoints, 1))
    return Trajectory(
        timestamps=timestamps,
        positions=base.copy(),
        velocities=base.copy() * 10,
        accelerations=base.copy() * 100,
        entity=group.config.name,
        joint_names=group.joint_names,
    )


class TestSplitTrajectoryMethod:
    def test_returns_one_trajectory_per_arm(self):
        group = make_group(2, 3)
        combined = make_combined(group)
        split = combined.split_trajectory(group)
        assert set(split.keys()) == {"left", "right"}

    def test_columns_sliced_in_arm_order(self):
        group = make_group(2, 3)
        combined = make_combined(group)
        split = combined.split_trajectory(group)
        np.testing.assert_array_equal(split["left"].positions[0], [0.0, 1.0])
        np.testing.assert_array_equal(split["right"].positions[0], [2.0, 3.0, 4.0])

    def test_velocities_and_accelerations_sliced_correctly(self):
        group = make_group(2, 2)
        combined = make_combined(group)
        split = combined.split_trajectory(group)
        np.testing.assert_array_equal(split["left"].velocities[0], [0.0, 10.0])
        np.testing.assert_array_equal(split["right"].velocities[0], [20.0, 30.0])
        np.testing.assert_array_equal(split["left"].accelerations[0], [0.0, 100.0])
        np.testing.assert_array_equal(split["right"].accelerations[0], [200.0, 300.0])

    def test_timestamps_shared_unchanged(self):
        group = make_group(2, 2)
        combined = make_combined(group, n_waypoints=7)
        split = combined.split_trajectory(group)
        np.testing.assert_array_equal(split["left"].timestamps, combined.timestamps)
        np.testing.assert_array_equal(split["right"].timestamps, combined.timestamps)
        assert split["left"].num_waypoints == split["right"].num_waypoints == 7

    def test_entity_tags_are_distinct_per_arm(self):
        # Regression test for the entity=self.entity bug -- must be
        # entity=arm_name so result.left.entity != result.right.entity.
        group = make_group(2, 2)
        combined = make_combined(group)
        split = combined.split_trajectory(group)
        assert split["left"].entity == "left"
        assert split["right"].entity == "right"
        assert len({split["left"].entity, split["right"].entity}) == 2

    def test_joint_names_are_per_arm_subset(self):
        group = make_group(2, 2)
        combined = make_combined(group)
        split = combined.split_trajectory(group)
        assert split["left"].joint_names == ["left_j0", "left_j1"]
        assert split["right"].joint_names == ["right_j0", "right_j1"]

    def test_uneven_dof_split(self):
        group = make_group(1, 5)
        combined = make_combined(group)
        split = combined.split_trajectory(group)
        np.testing.assert_array_equal(split["left"].positions[0], [0.0])
        np.testing.assert_array_equal(split["right"].positions[0], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_joint_names_none_skips_the_check(self):
        group = make_group(2, 2)
        combined = make_combined(group)
        combined.joint_names = None
        split = combined.split_trajectory(group)
        assert split["left"].dof == 2
        assert split["right"].joint_names is None

    def test_total_dof_mismatch_raises(self):
        group = make_group(2, 2)  # total dof = 4
        combined = Trajectory(
            timestamps=np.array([0.0, 1.0]),
            positions=np.zeros((2, 5)),  # 5 != 4
            velocities=np.zeros((2, 5)),
            accelerations=np.zeros((2, 5)),
        )
        with pytest.raises(ValueError, match="DOF"):
            combined.split_trajectory(group)

    # ---- the case that specifically targets the positional-order fix ----

    def test_swapped_order_WITHIN_one_arms_slice_raises(self):
        """Same two names, same arm, wrong order. A plain membership check
        (`name not in arm.config.joint_names`) would NOT catch this, since
        both names genuinely belong to 'left' -- only a positional
        equality check catches the swap."""
        group = make_group(2, 2)
        combined = make_combined(group)
        # left's slice is combined.joint_names[0:2] == ["left_j0", "left_j1"];
        # swap them in place.
        names = list(combined.joint_names)
        names[0], names[1] = names[1], names[0]
        combined.joint_names = names

        with pytest.raises(ValueError, match="left"):
            combined.split_trajectory(group)

    def test_name_that_truly_does_not_belong_still_raises(self):
        """Sanity check that the positional check still catches the
        original 'wrong arm entirely' case, not just swaps."""
        group = make_group(2, 2)
        combined = make_combined(group)
        names = list(combined.joint_names)
        names[0] = "not_a_real_joint"
        combined.joint_names = names

        with pytest.raises(ValueError):
            combined.split_trajectory(group)

    def test_correctly_ordered_names_do_not_raise(self):
        # Negative control: the happy path must NOT trip the new check.
        group = make_group(2, 2)
        combined = make_combined(group)
        split = combined.split_trajectory(group)  # should not raise
        assert split["left"].joint_names == ["left_j0", "left_j1"]
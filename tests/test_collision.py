# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for unified CollisionChecker.

Uses a minimal inline MuJoCo model with a 2-DOF arm and freejoint objects.
"""

import mujoco
import numpy as np
import pytest

from mj_manipulator.collision import CollisionChecker
from mj_manipulator.grasp_manager import GraspManager

# Minimal model: 2-joint arm that can collide with a table and objects
_COLLISION_XML = """
<mujoco model="test_collision">
  <worldbody>
    <!-- Table surface -->
    <geom name="table" type="box" size="1 1 0.01" pos="0 0 0.3" rgba="0.8 0.8 0.8 1"/>

    <!-- 2-DOF arm -->
    <body name="link1" pos="0 0 0.5">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom name="link1_geom" type="capsule" size="0.04" fromto="0 0 0 0.3 0 0"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1"/>
        <geom name="link2_geom" type="capsule" size="0.04" fromto="0 0 0 0.3 0 0"/>
        <body name="gripper/base" pos="0.3 0 0">
          <geom name="gripper_base_geom" type="box" size="0.02 0.04 0.02"/>
          <body name="gripper/left_finger" pos="0 0.03 0">
            <geom name="left_finger_geom" type="box" size="0.01 0.01 0.02"/>
          </body>
          <body name="gripper/right_finger" pos="0 -0.03 0">
            <geom name="right_finger_geom" type="box" size="0.01 0.01 0.02"/>
          </body>
        </body>
      </body>
    </body>

    <!-- Graspable object -->
    <body name="mug" pos="0.5 0 0.5">
      <joint name="mug_free" type="free"/>
      <geom name="mug_geom" type="cylinder" size="0.03 0.04"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model_and_data():
    model = mujoco.MjModel.from_xml_string(_COLLISION_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture
def joint_names():
    return ["joint1", "joint2"]


class TestCollisionCheckerSimpleMode:
    """Tests for CollisionChecker with no grasp awareness."""

    def test_constructs(self, model_and_data, joint_names):
        """Simple mode constructs without error."""
        model, data = model_and_data
        cc = CollisionChecker(model, data, joint_names)
        assert cc is not None

    def test_home_config(self, model_and_data, joint_names):
        """Home configuration validity depends on model geometry."""
        model, data = model_and_data
        cc = CollisionChecker(model, data, joint_names)
        q = np.zeros(2)
        # Just verify it runs without error
        result = cc.is_valid(q)
        assert isinstance(result, bool)

    def test_is_valid_batch(self, model_and_data, joint_names):
        """Batch checking works."""
        model, data = model_and_data
        cc = CollisionChecker(model, data, joint_names)
        qs = np.zeros((5, 2))
        results = cc.is_valid_batch(qs)
        assert results.shape == (5,)
        assert results.dtype == bool

    def test_invalid_joint_name_raises(self, model_and_data):
        """Non-existent joint name raises ValueError."""
        model, data = model_and_data
        with pytest.raises(ValueError, match="not found"):
            CollisionChecker(model, data, ["nonexistent_joint"])


class TestCollisionCheckerLiveMode:
    """Tests for CollisionChecker with live GraspManager."""

    def test_constructs_with_grasp_manager(self, model_and_data, joint_names):
        """Live mode constructs with GraspManager."""
        model, data = model_and_data
        gm = GraspManager(model, data)
        cc = CollisionChecker(model, data, joint_names, grasp_manager=gm)
        assert cc._grasp_manager is gm

    def test_uses_temp_data(self, model_and_data, joint_names):
        """Live mode creates temp data separate from live data."""
        model, data = model_and_data
        gm = GraspManager(model, data)
        cc = CollisionChecker(model, data, joint_names, grasp_manager=gm)
        assert cc.data is not data
        assert cc._live_data is data

    def test_is_valid_with_grasp(self, model_and_data, joint_names):
        """is_valid works when an object is grasped."""
        model, data = model_and_data
        gm = GraspManager(model, data)
        gm.mark_grasped("mug", "right")
        gm.attach_object("mug", "gripper/right_finger")

        cc = CollisionChecker(model, data, joint_names, grasp_manager=gm)
        q = np.zeros(2)
        result = cc.is_valid(q)
        assert isinstance(result, bool)


class TestCollisionCheckerSnapshotMode:
    """Tests for CollisionChecker with frozen grasp state."""

    def test_constructs_with_snapshot(self, model_and_data, joint_names):
        """Snapshot mode constructs with frozen state."""
        model, data = model_and_data
        cc = CollisionChecker(
            model,
            data,
            joint_names,
            grasped_objects=frozenset([("mug", "right")]),
            attachments={},
        )
        assert cc._grasp_manager is None

    def test_uses_provided_data(self, model_and_data, joint_names):
        """Snapshot mode uses provided data directly (for thread safety)."""
        model, data = model_and_data
        private_data = mujoco.MjData(model)
        cc = CollisionChecker(
            model,
            private_data,
            joint_names,
            grasped_objects=frozenset(),
            attachments={},
        )
        assert cc.data is private_data

    def test_is_valid_with_snapshot(self, model_and_data, joint_names):
        """is_valid works with snapshot grasp state."""
        model, data = model_and_data
        gm = GraspManager(model, data)
        gm.mark_grasped("mug", "right")
        gm.attach_object("mug", "gripper/right_finger")

        # Create snapshot
        grasped = frozenset(gm.grasped.items())
        attachments = dict(gm._attachments)

        private_data = mujoco.MjData(model)
        private_data.qpos[:] = data.qpos
        mujoco.mj_forward(model, private_data)

        cc = CollisionChecker(
            model,
            private_data,
            joint_names,
            grasped_objects=grasped,
            attachments=attachments,
        )
        q = np.zeros(2)
        result = cc.is_valid(q)
        assert isinstance(result, bool)

    def test_thread_safe_instances(self, model_and_data, joint_names):
        """Multiple snapshot instances don't share state."""
        model, data = model_and_data

        data1 = mujoco.MjData(model)
        data2 = mujoco.MjData(model)

        cc1 = CollisionChecker(
            model,
            data1,
            joint_names,
            grasped_objects=frozenset(),
            attachments={},
        )
        cc2 = CollisionChecker(
            model,
            data2,
            joint_names,
            grasped_objects=frozenset(),
            attachments={},
        )

        assert cc1.data is not cc2.data


class TestIsArmInCollision:
    """Tests for is_arm_in_collision (reactive cartesian control)."""

    def test_callable(self, model_and_data, joint_names):
        """is_arm_in_collision runs without error."""
        model, data = model_and_data
        gm = GraspManager(model, data)
        cc = CollisionChecker(model, data, joint_names, grasp_manager=gm)

        result = cc.is_arm_in_collision(q=np.zeros(2))
        assert isinstance(result, bool)

    def test_uses_current_state_when_no_q(self, model_and_data, joint_names):
        """When q=None, uses current data state."""
        model, data = model_and_data
        gm = GraspManager(model, data)
        cc = CollisionChecker(model, data, joint_names, grasp_manager=gm)

        result = cc.is_arm_in_collision(q=None)
        assert isinstance(result, bool)


class TestDebugContacts:
    """Tests for debug_contacts helper."""

    def test_callable(self, model_and_data, joint_names):
        """debug_contacts runs without error."""
        model, data = model_and_data
        gm = GraspManager(model, data)
        cc = CollisionChecker(model, data, joint_names, grasp_manager=gm)
        cc.debug_contacts(np.zeros(2))  # Should print, not raise


def _place_mug(model, data, pos):
    """Move the free-floating mug to ``pos`` (world frame)."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mug_free")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr : adr + 3] = pos
    data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)


class TestCollisionMargin:
    """margin_mm turns near misses into collisions without touching the shared model."""

    # Mug beside link2 at q=0: capsule radius 0.04 + cylinder radius 0.03,
    # centers 0.073 m apart in y -> a 3 mm gap.
    NEAR_MISS = [0.45, 0.073, 0.5]

    def test_near_miss_is_valid_without_margin(self, model_and_data, joint_names):
        model, data = model_and_data
        _place_mug(model, data, self.NEAR_MISS)
        cc = CollisionChecker(model, data, joint_names)
        assert cc.is_valid(np.zeros(2))

    def test_near_miss_is_invalid_inside_margin(self, model_and_data, joint_names):
        model, data = model_and_data
        _place_mug(model, data, self.NEAR_MISS)
        cc = CollisionChecker(model, data, joint_names, margin_mm=5.0)
        assert not cc.is_valid(np.zeros(2))

    def test_gap_beyond_margin_stays_valid(self, model_and_data, joint_names):
        model, data = model_and_data
        _place_mug(model, data, self.NEAR_MISS)
        cc = CollisionChecker(model, data, joint_names, margin_mm=1.0)
        assert cc.is_valid(np.zeros(2))

    def test_shared_model_is_not_mutated(self, model_and_data, joint_names):
        model, data = model_and_data
        before = model.geom_margin.copy()
        CollisionChecker(model, data, joint_names, margin_mm=5.0)
        np.testing.assert_array_equal(model.geom_margin, before)


class TestBodyIdsForJoints:
    """body_ids_for_joints returns each joint's body and its descendants."""

    def _ids(self, model, *names):
        return {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}

    def test_joint_includes_descendants(self, model_and_data, joint_names):
        model, data = model_and_data
        cc = CollisionChecker(model, data, joint_names)
        expected = self._ids(model, "link2", "gripper/base", "gripper/left_finger", "gripper/right_finger")
        assert cc.body_ids_for_joints(["joint2"]) == expected

    def test_extra_bodies_are_added(self, model_and_data, joint_names):
        model, data = model_and_data
        cc = CollisionChecker(model, data, joint_names)
        ids = cc.body_ids_for_joints(["joint2"], ["mug"])
        assert self._ids(model, "mug") <= ids

    def test_unknown_joint_raises(self, model_and_data, joint_names):
        model, data = model_and_data
        cc = CollisionChecker(model, data, joint_names)
        with pytest.raises(ValueError, match="not found"):
            cc.body_ids_for_joints(["no_such_joint"])

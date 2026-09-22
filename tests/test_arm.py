# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for the generic Arm class.

Uses real menagerie robot models (UR5e) to verify state queries,
forward kinematics, and motion planning.
"""

import mujoco
import numpy as np
import pytest
from mj_environment import Environment

from mj_manipulator.arm import Arm, ArmRobotModel
from mj_manipulator.config import ArmConfig, KinematicLimits

# UR5e constants
UR5E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
UR5E_HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
UR5E_VEL = np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28]) * 0.5
UR5E_ACC = np.array([2.5, 2.5, 2.5, 5.0, 5.0, 5.0]) * 0.5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ur5e_config() -> ArmConfig:
    return ArmConfig(
        name="ur5e",
        entity_type="arm",
        joint_names=UR5E_JOINTS,
        kinematic_limits=KinematicLimits(velocity=UR5E_VEL, acceleration=UR5E_ACC),
        ee_site="attachment_site",
    )


@pytest.fixture
def ur5e_env():
    """Create Environment with UR5e scene."""
    try:
        from mj_manipulator.menagerie import menagerie_scene

        scene = menagerie_scene("universal_robots_ur5e")
    except FileNotFoundError:
        pytest.skip("mujoco_menagerie not found")
    return Environment(str(scene))


@pytest.fixture
def ur5e_arm(ur5e_env):
    """Create Arm from UR5e environment."""
    config = _ur5e_config()
    arm = Arm(ur5e_env, config)

    # Set to home configuration
    for i, idx in enumerate(arm.joint_qpos_indices):
        ur5e_env.data.qpos[idx] = UR5E_HOME[i]
    mujoco.mj_forward(ur5e_env.model, ur5e_env.data)

    return arm


# ---------------------------------------------------------------------------
# State query tests
# ---------------------------------------------------------------------------


class TestStateQueries:
    """Tests for joint position/velocity and EE pose queries."""

    def test_arm_robot_model(self, ur5e_arm):
        """ArmRobotModel delegates to Arm correctly."""
        adapter = ArmRobotModel(ur5e_arm)
        assert adapter.dof == 6
        lower, _ = adapter.joint_limits
        assert len(lower) == 6

        pose = adapter.forward_kinematics(UR5E_HOME)
        assert pose.shape == (4, 4)

    def test_joint_positions(self, ur5e_arm):
        """get_joint_positions returns correct values after setting home."""
        q = ur5e_arm.get_joint_positions()
        np.testing.assert_allclose(q, UR5E_HOME, atol=1e-6)

    def test_joint_velocities_zero(self, ur5e_arm):
        """Velocities are zero at rest."""
        qd = ur5e_arm.get_joint_velocities()
        np.testing.assert_allclose(qd, 0, atol=1e-10)

    def test_dof(self, ur5e_arm):
        """UR5e has 6 DOF."""
        assert ur5e_arm.dof == 6

    def test_joint_limits(self, ur5e_arm):
        """Joint limits have correct shape."""
        lower, upper = ur5e_arm.get_joint_limits()
        assert len(lower) == 6
        assert len(upper) == 6
        assert np.all(lower < upper)

    def test_ee_pose_shape(self, ur5e_arm):
        """EE pose is a 4x4 homogeneous transform."""
        pose = ur5e_arm.get_ee_pose()
        assert pose.shape == (4, 4)
        # Valid rotation matrix: det = 1
        np.testing.assert_allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6)

    def test_actuator_discovery(self, ur5e_arm):
        """Actuator IDs are found for all joints."""
        assert len(ur5e_arm.actuator_ids) == 6


# ---------------------------------------------------------------------------
# Forward kinematics tests
# ---------------------------------------------------------------------------


class TestForwardKinematics:
    """Tests for non-destructive FK computation."""

    def test_fk_matches_live(self, ur5e_arm):
        """FK at current config matches live EE pose."""
        q = ur5e_arm.get_joint_positions()
        fk_pose = ur5e_arm.forward_kinematics(q)
        live_pose = ur5e_arm.get_ee_pose()
        np.testing.assert_allclose(fk_pose, live_pose, atol=1e-6)

    def test_fk_does_not_modify_state(self, ur5e_arm):
        """FK computation doesn't change live joint positions."""
        q_before = ur5e_arm.get_joint_positions().copy()

        # Compute FK at a different configuration
        q_other = UR5E_HOME + 0.1
        ur5e_arm.forward_kinematics(q_other)

        q_after = ur5e_arm.get_joint_positions()
        np.testing.assert_allclose(q_after, q_before, atol=1e-10)

    def test_fk_at_different_config(self, ur5e_arm):
        """FK at different config gives different pose."""
        pose1 = ur5e_arm.forward_kinematics(UR5E_HOME)
        pose2 = ur5e_arm.forward_kinematics(UR5E_HOME + 0.3)
        assert not np.allclose(pose1, pose2, atol=1e-3)

    def test_tcp_offset(self, ur5e_env):
        """tcp_offset shifts FK result by the given transform."""
        offset = np.eye(4)
        offset[2, 3] = 0.1  # 10cm along Z

        config_no_offset = _ur5e_config()
        config_with_offset = _ur5e_config()
        config_with_offset.tcp_offset = offset

        arm_no = Arm(ur5e_env, config_no_offset)
        arm_yes = Arm(ur5e_env, config_with_offset)

        # Set home
        for i, idx in enumerate(arm_no.joint_qpos_indices):
            ur5e_env.data.qpos[idx] = UR5E_HOME[i]
        mujoco.mj_forward(ur5e_env.model, ur5e_env.data)

        pose_no = arm_no.forward_kinematics(UR5E_HOME)
        pose_yes = arm_yes.forward_kinematics(UR5E_HOME)

        expected = pose_no @ offset
        np.testing.assert_allclose(pose_yes, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrors:
    """Tests for error handling."""

    def test_invalid_joint_name(self, ur5e_env):
        """ValueError for non-existent joint name."""
        config = ArmConfig(
            name="bad",
            entity_type="arm",
            joint_names=["nonexistent_joint"],
            kinematic_limits=KinematicLimits(
                velocity=np.array([1.0]),
                acceleration=np.array([1.0]),
            ),
        )
        with pytest.raises(ValueError, match="not found in model"):
            Arm(ur5e_env, config)

    def test_invalid_ee_site(self, ur5e_env):
        """ValueError for non-existent EE site."""
        config = ArmConfig(
            name="bad",
            entity_type="arm",
            joint_names=UR5E_JOINTS,
            kinematic_limits=KinematicLimits(velocity=UR5E_VEL, acceleration=UR5E_ACC),
            ee_site="nonexistent_site",
        )
        with pytest.raises(ValueError, match="not found in model"):
            Arm(ur5e_env, config)

    def test_no_ee_site_raises_on_fk(self, ur5e_env):
        """RuntimeError when calling FK without ee_site."""
        config = ArmConfig(
            name="no_ee",
            entity_type="arm",
            joint_names=UR5E_JOINTS,
            kinematic_limits=KinematicLimits(velocity=UR5E_VEL, acceleration=UR5E_ACC),
            ee_site="",
        )
        arm = Arm(ur5e_env, config)
        with pytest.raises(RuntimeError, match="No ee_site"):
            arm.get_ee_pose()

"""Unit tests for mj_manipulator.arm_group.ArmGroup.

Scope: these tests exercise ArmGroup's own logic — the Mapping protocol,
state-query concatenation, frame normalization, cross-product goal
filtering, and planner-config construction — using a real Arm backed by a
small synthetic-but-real MuJoCo model/Environment, with a stubbed-out
CBiRRT planner and collision checker. The planner/collision boundary is
still faked (FakePlanner/FakeCollision) since exercising the actual
CBiRRT planner belongs in an integration test (e.g. the acceptance test in
openarm/tests/test_openarm_bimanual_planning.py), not here.

Run with:  uv run pytest mj_manipulator/tests/test_arm_group.py -v
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from mj_environment import Environment

from mj_manipulator.arm import Arm
from mj_manipulator.arm_group import ArmGroup, ContextRobotModel
from mj_manipulator.config import ArmConfig, ArmGroupConfig, KinematicLimits, PlanningDefaults

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
# Adapter tests
# ---------------------------------------------------------------------------


class TestAdapters:
    """Tests for pycbirrt RobotModel adapters."""

    def test_context_robot_model(self, ur5e_arm):
        """ContextRobotModel gives same FK as Arm."""
        model = ur5e_arm.env.model
        data = mujoco.MjData(model)
        np.copyto(data.qpos, ur5e_arm.env.data.qpos)
        mujoco.mj_forward(model, data)

        ctx_model = ContextRobotModel(
            model=model,
            data=data,
            joint_qpos_indices=ur5e_arm.joint_qpos_indices,
            ee_site_id_group=[ur5e_arm.ee_site_id],
            joint_limits=ur5e_arm.get_joint_limits(),
        )

        assert ctx_model.dof == 6
        pose_arm = ur5e_arm.forward_kinematics(UR5E_HOME)
        pose_ctx = ctx_model.forward_kinematics(UR5E_HOME)
        np.testing.assert_allclose(pose_arm, pose_ctx, atol=1e-6)

    def test_context_model_isolation(self, ur5e_arm):
        """ContextRobotModel FK doesn't affect live env."""
        q_before = ur5e_arm.get_joint_positions().copy()

        model = ur5e_arm.env.model
        data = mujoco.MjData(model)

        ctx_model = ContextRobotModel(
            model=model,
            data=data,
            joint_qpos_indices=ur5e_arm.joint_qpos_indices,
            ee_site_id_group=[ur5e_arm.ee_site_id],
            joint_limits=ur5e_arm.get_joint_limits(),
        )

        # FK at a wildly different config
        ctx_model.forward_kinematics(UR5E_HOME + 1.0)

        q_after = ur5e_arm.get_joint_positions()
        np.testing.assert_allclose(q_after, q_before, atol=1e-10)

# =============================================================================
# Fakes
# =============================================================================


class FakeIKSolver:
    """Returns a fixed, caller-controlled list of IK solutions."""

    def __init__(self, solutions: list[np.ndarray] | None = None):
        self.solutions = solutions if solutions is not None else []
        self.calls: list[np.ndarray] = []

    def solve_valid(self, pose: np.ndarray) -> list[np.ndarray]:
        self.calls.append(pose)
        return list(self.solutions)


def _chain_xml(name: str, dof: int, y: float) -> str:
    """MJCF for one hinge-joint kinematic chain, joints named f"{name}_j{i}".

    Joint range is +-3.14 rad -- wide enough that arbitrary test joint
    values never trip Arm.set_joint_positions' limit check, and narrow
    enough that _detect_angular_joints() still classifies these as
    non-continuous by default (its threshold is 2*pi*1.5).
    """
    parts = []
    for i in range(dof):
        pos = f"0 {y} 0.5" if i == 0 else "0.2 0 0"
        parts.append(f'<body name="{name}_link{i}" pos="{pos}" gravcomp="1">')
        parts.append(
            f'<joint name="{name}_j{i}" type="hinge" axis="0 0 1" '
            'range="-3.14 3.14" limited="true"/>'
            '<geom type="capsule" size="0.03" fromto="0 0 0 0.2 0 0"/>'
        )
    parts.append("</body>" * dof)
    return "".join(parts)


def build_env(arm_dofs: dict[str, int]) -> Environment:
    """Compile a minimal real MuJoCo Environment with one chain per arm.

    Used instead of a full robot model so these tests stay fast and
    independent of any specific robot asset, while still exercising a real
    mujoco.MjModel/MjData and mj_environment.Environment underneath Arm.
    """
    bodies = "".join(_chain_xml(name, dof, y=0.3 * i) for i, (name, dof) in enumerate(arm_dofs.items()))
    # angle="radian": MuJoCo's compiler default is degrees, which would
    # silently reinterpret the "-3.14 3.14" joint ranges below as +-3.14
    # degrees instead of +-pi radians.
    xml = f'<mujoco model="test_arm_group"><compiler angle="radian"/><worldbody>{bodies}</worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    return Environment.from_model(model)


def make_arm(
    name: str,
    dof: int,
    *,
    env: Environment | None = None,
    ik_solver=None,
    grasp_manager=None,
) -> Arm:
    """Build a real mj_manipulator.arm.Arm backed by a real Environment.

    Builds a fresh single-arm Environment unless one is given, so tests
    that need arms to share (or conflict on) an Environment can pass one
    in explicitly.
    """
    if env is None:
        env = build_env({name: dof})
    config = ArmConfig(
        name=name,
        entity_type="arm",
        joint_names=[f"{name}_j{i}" for i in range(dof)],
        kinematic_limits=KinematicLimits(
            velocity=np.full(dof, 1.0),
            acceleration=np.full(dof, 2.0),
        ),
    )
    return Arm(env, config, ik_solver=ik_solver, grasp_manager=grasp_manager)


class FakeCollision:
    """Controls which combined configs 'pass' collision checking.

    invalid_if(q) -> bool decides rejection; default accepts everything.
    """

    def __init__(self, invalid_if=None):
        self._invalid_if = invalid_if or (lambda q: False)
        self.checked: list[np.ndarray] = []

    def is_valid(self, q: np.ndarray) -> bool:
        self.checked.append(q)
        return not self._invalid_if(q)

    def get_contacts(self, q: np.ndarray):
        return []


class FakePlanner:
    """Stand-in for pycbirrt.CBiRRT — only .collision and .plan() are used
    by the frame-sequence machinery under test."""

    def __init__(self, collision: FakeCollision, path_by_call=None, raise_on_call=None):
        self.collision = collision
        # path_by_call: list of paths to return, one per .plan() call
        self._paths = list(path_by_call) if path_by_call is not None else None
        self._raise = raise_on_call or {}
        self.plan_calls: list[dict] = []

    def plan(self, start, goal, seed=None):
        call_idx = len(self.plan_calls)
        self.plan_calls.append({"start": start, "goal": goal, "seed": seed})
        if call_idx in self._raise:
            raise self._raise[call_idx]
        if self._paths is None:
            # default: straight line from start to the first goal candidate
            return [start, goal[0]]
        return self._paths[call_idx]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def two_arm_group():
    """A left/right ArmGroup, 2 DOF each, no IK solver, no grasp_manager,
    backed by one real (minimal) MuJoCo Environment shared by both arms so
    the real planning path (_detect_angular_joints, etc.) is exercisable.
    """
    env = build_env({"left": 2, "right": 2})
    left = make_arm("left", 2, env=env)
    right = make_arm("right", 2, env=env)
    arms = {"left": left, "right": right}
    config = ArmGroupConfig(
        name="bimanual",
        entity_type="arm_group",
        joint_names=left.config.joint_names + right.config.joint_names,
        max_bimanual_IK_solutions=20,
        planning_defaults=PlanningDefaults(),
    )
    return ArmGroup(arms, config)



# =============================================================================
# Construction
# =============================================================================


class TestConstruction:
    def test_dof_is_sum_of_arm_dofs(self, two_arm_group):
        assert two_arm_group.dof == 4

    def test_joint_names_mismatch_raises(self):
        env = build_env({"left": 2, "right": 2})
        left = make_arm("left", 2, env=env)
        right = make_arm("right", 2, env=env)
        bad_config = ArmGroupConfig(
            name="bimanual",
            entity_type="arm_group",
            joint_names=["wrong_name_a", "wrong_name_b"],
        )
        with pytest.raises(ValueError, match="joint_names mismatch"):
            ArmGroup({"left": left, "right": right}, bad_config)


# =============================================================================
# Mapping protocol (item 0.1)
# =============================================================================


class TestMappingProtocol:
    def test_len(self, two_arm_group):
        assert len(two_arm_group) == 2

    def test_getitem(self, two_arm_group):
        assert two_arm_group["left"].config.name == "left"
        assert two_arm_group["right"].config.name == "right"

    def test_getitem_missing_raises_keyerror(self, two_arm_group):
        with pytest.raises(KeyError):
            two_arm_group["center"]

    def test_iter_yields_keys_in_insertion_order(self, two_arm_group):
        assert list(two_arm_group) == ["left", "right"]

    def test_contains(self, two_arm_group):
        assert "left" in two_arm_group
        assert "center" not in two_arm_group

    def test_keys_values_items(self, two_arm_group):
        assert set(two_arm_group.keys()) == {"left", "right"}
        assert {a.config.name for a in two_arm_group.values()} == {"left", "right"}
        assert {k: v.config.name for k, v in two_arm_group.items()} == {
            "left": "left",
            "right": "right",
        }

    def test_get_with_default(self, two_arm_group):
        assert two_arm_group.get("left") is two_arm_group["left"]
        assert two_arm_group.get("center") is None
        assert two_arm_group.get("center", "fallback") == "fallback"

    def test_arm_method_matches_getitem(self, two_arm_group):
        assert two_arm_group.arm("left") is two_arm_group["left"]

    def test_arm_method_raises_valueerror_not_keyerror(self, two_arm_group):
        # .arm() is the friendlier, REPL-facing accessor — keep its
        # ValueError contract distinct from __getitem__'s KeyError.
        with pytest.raises(ValueError, match="not found in this group"):
            two_arm_group.arm("center")


# =============================================================================
# State-query concatenation
# =============================================================================


class TestStateQueries:
    def test_joint_names_concatenated_in_arm_order(self, two_arm_group):
        assert two_arm_group.joint_names == [
            "left_j0", "left_j1", "right_j0", "right_j1",
        ]

    def test_kinematic_limits_concatenated(self, two_arm_group):
        limits = two_arm_group.kinematic_limits
        assert limits.velocity.shape == (4,)
        assert limits.acceleration.shape == (4,)
        np.testing.assert_array_equal(limits.velocity, [1.0, 1.0, 1.0, 1.0])

    def test_get_joint_limits_returns_tuple_not_list(self, two_arm_group):
        # Regression test for item 0.5: the annotation used to claim
        # list[tuple[...]] but the implementation always returned a single
        # (lower, upper) tuple of concatenated arrays. Lock in the real
        # behavior so the annotation can never silently drift again.
        result = two_arm_group.get_joint_limits()
        assert isinstance(result, tuple)
        assert len(result) == 2
        lo, hi = result
        assert lo.shape == (4,)
        assert hi.shape == (4,)

    def test_get_set_joint_positions_round_trip(self, two_arm_group):
        q = np.array([0.1, 0.2, 0.3, 0.4])
        two_arm_group.set_joint_positions(q)
        np.testing.assert_array_equal(two_arm_group.get_joint_positions(), q)
        # and it actually landed on the correct per-arm sub-arrays
        np.testing.assert_array_equal(two_arm_group["left"].get_joint_positions(), [0.1, 0.2])
        np.testing.assert_array_equal(two_arm_group["right"].get_joint_positions(), [0.3, 0.4])

    def test_set_joint_positions_wrong_length_raises(self, two_arm_group):
        with pytest.raises(ValueError, match="Expected 4 joints"):
            two_arm_group.set_joint_positions(np.zeros(3))

    def test_split_by_arm(self, two_arm_group):
        q = np.array([1.0, 2.0, 3.0, 4.0])
        split = two_arm_group._split_by_arm(q)
        np.testing.assert_array_equal(split["left"], [1.0, 2.0])
        np.testing.assert_array_equal(split["right"], [3.0, 4.0])

    def test_env_raises_when_no_arm_has_one(self):
        left = make_arm("left", 2)
        right = make_arm("right", 2)
        left.env = None
        right.env = None
        config = ArmGroupConfig(
            name="g", entity_type="arm_group",
            joint_names=left.config.joint_names + right.config.joint_names,
        )
        group = ArmGroup({"left": left, "right": right}, config)
        with pytest.raises(ValueError, match="No arms in this group have an associated Environment"):
            _ = group.env

    def test_env_resolves_when_single_shared_env(self):
        shared_env = build_env({"left": 1, "right": 1})
        left = make_arm("left", 1, env=shared_env)
        right = make_arm("right", 1, env=shared_env)
        config = ArmGroupConfig(
            name="g", entity_type="arm_group",
            joint_names=left.config.joint_names + right.config.joint_names,
        )
        group = ArmGroup({"left": left, "right": right}, config)
        assert group.env is shared_env

    def test_env_raises_on_conflicting_envs(self):
        left = make_arm("left", 1)
        right = make_arm("right", 1)
        config = ArmGroupConfig(
            name="g", entity_type="arm_group",
            joint_names=left.config.joint_names + right.config.joint_names,
        )
        group = ArmGroup({"left": left, "right": right}, config)
        with pytest.raises(ValueError, match="Multiple Environments"):
            group.env

    def test_grasp_manager_none_when_no_arm_has_one(self, two_arm_group):
        assert two_arm_group.grasp_manager is None

    def test_grasp_manager_raises_on_multiple_distinct_managers(self):
        left = make_arm("left", 1, grasp_manager=object())
        right = make_arm("right", 1, grasp_manager=object())
        config = ArmGroupConfig(
            name="g", entity_type="arm_group",
            joint_names=left.config.joint_names + right.config.joint_names,
        )
        group = ArmGroup({"left": left, "right": right}, config)
        with pytest.raises(ValueError, match="Multiple GraspManagers"):
            _ = group.grasp_manager


# =============================================================================
# _as_frames — targets normalization
# =============================================================================


class TestAsFrames:
    def test_empty_targets(self):
        assert ArmGroup._as_frames({}) == []

    def test_bare_targets_become_single_frame(self):
        frames = ArmGroup._as_frames({"left": "L", "right": "R"})
        assert frames == [{"left": "L", "right": "R"}]

    def test_list_targets_become_sequence_of_frames(self):
        frames = ArmGroup._as_frames({"left": ["L0", "L1", "L2"]})
        assert frames == [{"left": "L0"}, {"left": "L1"}, {"left": "L2"}]

    def test_length_1_arm_broadcasts_across_sequence(self):
        frames = ArmGroup._as_frames({"left": ["L0", "L1"], "right": "R_hold"})
        assert frames == [
            {"left": "L0", "right": "R_hold"},
            {"left": "L1", "right": "R_hold"},
        ]

    def test_mismatched_sequence_lengths_raise(self):
        with pytest.raises(ValueError, match="must all be 1"):
            ArmGroup._as_frames({"left": ["L0", "L1"], "right": ["R0", "R1", "R2"]})


# =============================================================================
# _combined_goals_for_frame — cross-product + collision filtering
# =============================================================================


class TestCombinedGoalsForFrame:
    def test_unknown_arm_in_frame_raises(self, two_arm_group):
        planner = FakePlanner(FakeCollision())
        with pytest.raises(ValueError, match="unknown arm"):
            two_arm_group._combined_goals_for_frame(
                {"center": "X"}, two_arm_group.get_joint_positions(),
                lambda arm, target, q_ref: [np.zeros(arm.dof)], planner,
            )

    def test_unnamed_arm_holds_at_running_config(self, two_arm_group):
        q_start = np.array([9.0, 9.0, 0.0, 0.0])  # left already at (9, 9)
        planner = FakePlanner(FakeCollision())

        def resolve(arm, target, q_ref):
            return [np.array([target, target])]

        goals = two_arm_group._combined_goals_for_frame(
            {"right": 5.0}, q_start, resolve, planner,
        )
        assert goals is not None
        assert len(goals) == 1
        np.testing.assert_array_equal(goals[0], [9.0, 9.0, 5.0, 5.0])

    def test_no_candidates_for_named_arm_returns_none(self, two_arm_group):
        planner = FakePlanner(FakeCollision())
        goals = two_arm_group._combined_goals_for_frame(
            {"left": "unreachable"}, two_arm_group.get_joint_positions(),
            lambda arm, target, q_ref: [], planner,
        )
        assert goals is None

    def test_cross_product_filtered_by_collision(self, two_arm_group):
        # left has 2 candidates, right has 2 candidates -> 4 combos.
        # Reject any combo where left's first joint equals right's first
        # joint, to prove the filter runs on the COMBINED vector.
        def invalid_if(q):
            return q[0] == q[2]

        planner = FakePlanner(FakeCollision(invalid_if=invalid_if))

        def resolve(arm, target, q_ref):
            if arm.config.name == "left":
                return [np.array([1.0, 1.0]), np.array([2.0, 2.0])]
            return [np.array([1.0, 1.0]), np.array([3.0, 3.0])]

        goals = two_arm_group._combined_goals_for_frame(
            {"left": "L", "right": "R"}, two_arm_group.get_joint_positions(),
            resolve, planner,
        )
        assert goals is not None
        # (left=1, right=1) is rejected (1==1); the other three survive.
        assert len(goals) == 3
        rejected = np.array([1.0, 1.0, 1.0, 1.0])
        assert not any(np.array_equal(g, rejected) for g in goals)

    def test_respects_max_bimanual_ik_solutions_cap(self):
        left = make_arm("left", 1)
        right = make_arm("right", 1)
        config = ArmGroupConfig(
            name="g", entity_type="arm_group",
            joint_names=left.config.joint_names + right.config.joint_names,
            max_bimanual_IK_solutions=2,
        )
        group = ArmGroup({"left": left, "right": right}, config)
        planner = FakePlanner(FakeCollision())  # accept everything

        def resolve(arm, target, q_ref):
            # 5 candidates each -> 25 combos, but should stop at cap=2
            return [np.array([float(i)]) for i in range(5)]

        goals = group._combined_goals_for_frame(
            {"left": "L", "right": "R"}, group.get_joint_positions(), resolve, planner,
        )
        assert len(goals) == 2

    def test_all_combos_invalid_returns_none(self, two_arm_group):
        planner = FakePlanner(FakeCollision(invalid_if=lambda q: True))

        def resolve(arm, target, q_ref):
            return [np.zeros(arm.dof)]

        goals = two_arm_group._combined_goals_for_frame(
            {"left": "L", "right": "R"}, two_arm_group.get_joint_positions(),
            resolve, planner,
        )
        assert goals is None


# =============================================================================
# _plan_frame_sequence — multi-frame sequencing over a stubbed planner
# =============================================================================


class TestPlanFrameSequence:
    def test_constraint_tsrs_not_implemented(self, two_arm_group, monkeypatch):
        monkeypatch.setattr(
            two_arm_group, "create_planner", lambda config=None: FakePlanner(FakeCollision())
        )
        with pytest.raises(NotImplementedError):
            two_arm_group._plan_frame_sequence(
                {"left": "L"}, lambda arm, t, q: [np.zeros(arm.dof)],
                constraint_tsrs=["fake_tsr"],
            )

    def test_empty_targets_returns_none(self, two_arm_group, monkeypatch):
        monkeypatch.setattr(
            two_arm_group, "create_planner", lambda config=None: FakePlanner(FakeCollision())
        )
        assert two_arm_group._plan_frame_sequence({}, lambda a, t, q: [q]) is None

    def test_single_frame_success_returns_full_path(self, two_arm_group, monkeypatch):
        planner = FakePlanner(FakeCollision())
        monkeypatch.setattr(two_arm_group, "create_planner", lambda config=None: planner)

        def resolve(arm, target, q_ref):
            return [np.full(arm.dof, target)]

        path = two_arm_group._plan_frame_sequence({"left": 1.0, "right": 2.0}, resolve)
        assert path is not None
        assert len(path) == 2  # start + one planned segment endpoint
        np.testing.assert_array_equal(path[0], two_arm_group.get_joint_positions())

    def test_no_collision_free_goal_stops_planning(self, two_arm_group, monkeypatch):
        planner = FakePlanner(FakeCollision(invalid_if=lambda q: True))
        monkeypatch.setattr(two_arm_group, "create_planner", lambda config=None: planner)

        def resolve(arm, target, q_ref):
            return [np.full(arm.dof, target)]

        result = two_arm_group._plan_frame_sequence({"left": 1.0}, resolve)
        assert result is None
        assert len(planner.plan_calls) == 0  # never even attempted planner.plan()

    def test_planner_raises_planning_error_returns_none(self, two_arm_group, monkeypatch):
        from pycbirrt.exceptions import PlanningError

        planner = FakePlanner(FakeCollision(), raise_on_call={0: PlanningError("no path")})
        monkeypatch.setattr(two_arm_group, "create_planner", lambda config=None: planner)

        def resolve(arm, target, q_ref):
            return [np.full(arm.dof, target)]

        assert two_arm_group._plan_frame_sequence({"left": 1.0}, resolve) is None

    def test_planner_returns_none_segment_stops(self, two_arm_group, monkeypatch):
        planner = FakePlanner(FakeCollision(), path_by_call=[None])
        monkeypatch.setattr(two_arm_group, "create_planner", lambda config=None: planner)

        def resolve(arm, target, q_ref):
            return [np.full(arm.dof, target)]

        assert two_arm_group._plan_frame_sequence({"left": 1.0}, resolve) is None

    def test_multi_frame_sequence_chains_segments(self, two_arm_group, monkeypatch):
        # Two frames -> two planner.plan() calls; each segment's start is
        # the previous segment's end (dedup of the boundary waypoint).
        seg0 = [np.zeros(4), np.array([1.0, 1.0, 0.0, 0.0])]
        seg1 = [np.array([1.0, 1.0, 0.0, 0.0]), np.array([1.0, 1.0, 2.0, 2.0])]
        planner = FakePlanner(FakeCollision(), path_by_call=[seg0, seg1])
        monkeypatch.setattr(two_arm_group, "create_planner", lambda config=None: planner)

        def resolve(arm, target, q_ref):
            return [np.full(arm.dof, target)]

        path = two_arm_group._plan_frame_sequence(
            {"left": [1.0], "right": [0.0, 2.0]}, resolve,
        )
        assert path is not None
        # start + seg0's non-duplicate tail + seg1's non-duplicate tail
        assert len(path) == 3
        np.testing.assert_array_equal(path[-1], [1.0, 1.0, 2.0, 2.0])
        assert len(planner.plan_calls) == 2

# =============================================================================
# test planner failure handling through the public API
# =============================================================================

class TestPublicPlanningAPIFailureHandling:
    """Planner exceptions raised inside .plan() surface as None through
    the group's public API — the group-level equivalent of the deleted
    per-arm TestPlannerFailureReturnsNone."""

    def test_plan_to_configuration_returns_none_on_planning_error(self, two_arm_group, monkeypatch):
        from pycbirrt.exceptions import AllStartConfigurationsInCollision

        planner = FakePlanner(FakeCollision(), raise_on_call={0: AllStartConfigurationsInCollision(1)})
        monkeypatch.setattr(two_arm_group, "create_planner", lambda config=None: planner)

        goal = {"left": np.array([0.5, 0.5]), "right": np.array([0.6, 0.6])}
        assert two_arm_group.plan_to_configuration(goal) is None

    def test_plan_to_configuration_returns_none_on_value_error(self, two_arm_group, monkeypatch):
        planner = FakePlanner(
            FakeCollision(), raise_on_call={0: ValueError("No valid start configurations available")}
        )
        monkeypatch.setattr(two_arm_group, "create_planner", lambda config=None: planner)

        goal = {"left": np.array([0.5, 0.5]), "right": np.array([0.6, 0.6])}
        assert two_arm_group.plan_to_configuration(goal) is None

    def test_plan_to_tsrs_returns_none_on_planning_error(self, two_arm_group, monkeypatch):
        from pycbirrt.exceptions import AllGoalConfigurationsInvalid
        from tsr.tsr import TSR

        planner = FakePlanner(FakeCollision(), raise_on_call={0: AllGoalConfigurationsInvalid(5)})
        monkeypatch.setattr(two_arm_group, "create_planner", lambda config=None: planner)
        two_arm_group["left"].ik_solver = FakeIKSolver(solutions=[np.zeros(2)])

        tsr = TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=np.zeros((6, 2)))
        assert two_arm_group.plan_to_tsrs({"left": tsr}) is None


# =============================================================================
# Candidate resolvers
# =============================================================================


class TestCandidateResolvers:
    def test_config_candidates_wraps_single_array(self, two_arm_group):
        arm = two_arm_group["left"]
        q = np.array([0.5, 0.5])
        result = ArmGroup._config_candidates(arm, q, q_ref=np.zeros(2))
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], q)

    def test_config_candidates_wrong_shape_raises(self, two_arm_group):
        arm = two_arm_group["left"]  # dof=2
        with pytest.raises(ValueError, match="expects 2 joints"):
            ArmGroup._config_candidates(arm, np.zeros(3), q_ref=np.zeros(2))

    def test_pose_candidates_requires_ik_solver(self, two_arm_group):
        arm = two_arm_group["left"]  # ik_solver is None
        with pytest.raises(RuntimeError, match="requires an IK solver"):
            two_arm_group._pose_candidates(arm, pose=np.eye(4), q_ref=np.zeros(2))

    def test_pose_candidates_empty_solutions_returns_empty(self, two_arm_group):
        arm = two_arm_group["left"]
        arm.ik_solver = FakeIKSolver(solutions=[])
        result = two_arm_group._pose_candidates(arm, pose=np.eye(4), q_ref=np.zeros(2))
        assert result == []

    def test_pose_candidates_sorted_nearest_first_and_truncated(self, two_arm_group):
        arm = two_arm_group["left"]
        arm.config.max_ik_solutions = 2
        far = np.array([10.0, 10.0])
        near = np.array([0.1, 0.1])
        mid = np.array([1.0, 1.0])
        arm.ik_solver = FakeIKSolver(solutions=[far, near, mid])

        result = two_arm_group._pose_candidates(arm, pose=np.eye(4), q_ref=np.zeros(2))
        assert len(result) == 2  # truncated to max_ik_solutions
        np.testing.assert_array_equal(result[0], near)
        np.testing.assert_array_equal(result[1], mid)

    def test_tsr_candidates_requires_ik_solver(self, two_arm_group):
        arm = two_arm_group["left"]

        class FakeTSR:
            def sample(self):
                return np.eye(4)

        with pytest.raises(RuntimeError, match="TSR planning requires"):
            two_arm_group._tsr_candidates(arm, FakeTSR(), q_ref=np.zeros(2), samples=3)

    def test_tsr_candidates_samples_n_times(self, two_arm_group):
        arm = two_arm_group["left"]
        arm.ik_solver = FakeIKSolver(solutions=[np.array([0.1, 0.1])])

        class FakeTSR:
            def sample(self):
                return np.eye(4)

        two_arm_group._tsr_candidates(arm, FakeTSR(), q_ref=np.zeros(2), samples=4)
        assert len(arm.ik_solver.calls) == 4


# =============================================================================
# Public planning API wiring (plan_to_configuration / plan_to_poses / plan_to_tsrs)
# =============================================================================


class TestPublicPlanningAPI:
    def test_plan_to_configuration_uses_config_candidates(self, two_arm_group, monkeypatch):
        planner = FakePlanner(FakeCollision())
        monkeypatch.setattr(two_arm_group, "create_planner", lambda config=None: planner)

        goal = {"left": np.array([0.5, 0.5]), "right": np.array([0.6, 0.6])}
        path = two_arm_group.plan_to_configuration(goal)
        assert path is not None

    def test_plan_to_configuration_rejects_wrong_shape(self, two_arm_group, monkeypatch):
        planner = FakePlanner(FakeCollision())
        monkeypatch.setattr(two_arm_group, "create_planner", lambda config=None: planner)

        # left is dof=2; giving it a length-3 vector surfaces as a
        # ValueError from _config_candidates, propagated up through
        # resolve_candidates rather than swallowed as a silent None.
        with pytest.raises(ValueError):
            two_arm_group.plan_to_configuration({"left": np.zeros(3)})

    def test_plan_to_poses_requires_ik_on_named_arm(self, two_arm_group, monkeypatch):
        planner = FakePlanner(FakeCollision())
        monkeypatch.setattr(two_arm_group, "create_planner", lambda config=None: planner)
        with pytest.raises(RuntimeError, match="requires an IK solver"):
            two_arm_group.plan_to_poses({"left": np.eye(4)})

    def test_plan_to_tsrs_requires_ik_on_named_arm(self, two_arm_group, monkeypatch):
        planner = FakePlanner(FakeCollision())
        monkeypatch.setattr(two_arm_group, "create_planner", lambda config=None: planner)

        class FakeTSR:
            def sample(self):
                return np.eye(4)

        with pytest.raises(RuntimeError, match="TSR planning requires"):
            two_arm_group.plan_to_tsrs({"left": FakeTSR()})


# =============================================================================
# _make_planner_config and _detect_angular_joints
# =============================================================================


class TestMakePlannerConfig:
    def test_no_override_builds_from_defaults(self, two_arm_group, monkeypatch):
        monkeypatch.setattr(two_arm_group, "_detect_angular_joints", lambda: None)
        cfg = two_arm_group._make_planner_config(timeout=None, planner_config=None)
        d = two_arm_group.config.planning_defaults
        assert cfg.timeout == d.timeout
        assert cfg.max_iterations == d.max_iterations
        assert cfg.step_size == d.step_size
        assert cfg.goal_bias == d.goal_bias
        assert cfg.smoothing_iterations == d.smoothing_iterations
        assert cfg.angular_joints is None
        assert cfg.abort_fn is None

    def test_timeout_override_without_explicit_planner_config(self, two_arm_group, monkeypatch):
        monkeypatch.setattr(two_arm_group, "_detect_angular_joints", lambda: None)
        cfg = two_arm_group._make_planner_config(timeout=1.23, planner_config=None)
        assert cfg.timeout == 1.23

    def test_angular_joints_wired_through(self, two_arm_group, monkeypatch):
        monkeypatch.setattr(two_arm_group, "_detect_angular_joints", lambda: (True, False, True, False))
        cfg = two_arm_group._make_planner_config(timeout=None, planner_config=None)
        assert cfg.angular_joints == (True, False, True, False)

    def test_abort_fn_wired_through(self, two_arm_group, monkeypatch):
        monkeypatch.setattr(two_arm_group, "_detect_angular_joints", lambda: None)
        fn = lambda: False
        cfg = two_arm_group._make_planner_config(timeout=None, planner_config=None, abort_fn=fn)
        assert cfg.abort_fn is fn

    def test_explicit_planner_config_passed_through_untouched(self, two_arm_group):
        from pycbirrt import CBiRRTConfig

        explicit = CBiRRTConfig(
            timeout=99.0, max_iterations=1, step_size=0.01,
            goal_bias=0.9, smoothing_iterations=0,
        )
        cfg = two_arm_group._make_planner_config(timeout=None, planner_config=explicit)
        assert cfg is explicit

    def test_explicit_planner_config_gets_timeout_override(self, two_arm_group):
        from pycbirrt import CBiRRTConfig

        explicit = CBiRRTConfig(
            timeout=99.0, max_iterations=1, step_size=0.01,
            goal_bias=0.9, smoothing_iterations=0,
        )
        cfg = two_arm_group._make_planner_config(timeout=5.0, planner_config=explicit)
        assert cfg is not explicit  # dataclasses.replace returns a new instance
        assert cfg.timeout == 5.0
        assert cfg.max_iterations == 1  # everything else preserved


class TestDetectAngularJoints:
    """Exercises _detect_angular_joints() against the real MuJoCo model
    backing two_arm_group, mutating jnt_limited/jnt_range in place on that
    real model to trigger each detection branch."""

    def test_no_continuous_joints_returns_none(self, two_arm_group):
        # Default fixture joints are all limited to +-3.14 rad -- well
        # under the 2*pi*1.5 continuous threshold.
        assert two_arm_group._detect_angular_joints() is None

    def test_unlimited_joint_detected_as_angular(self, two_arm_group):
        model = two_arm_group.env.model
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_j1")
        model.jnt_limited[jid] = 0  # joint 1 unlimited

        result = two_arm_group._detect_angular_joints()
        assert result == (False, True, False, False)

    def test_wide_range_joint_detected_as_angular(self, two_arm_group):
        model = two_arm_group.env.model
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_j0")
        model.jnt_range[jid] = [-20.0, 20.0]  # joint 2 range > 2*pi*1.5 -> continuous

        result = two_arm_group._detect_angular_joints()
        assert result == (False, False, True, False)


# =============================================================================
# check_collisions
# =============================================================================


class TestCheckCollisions:
    def test_unknown_arm_name_raises(self, two_arm_group, monkeypatch):
        monkeypatch.setattr(
            two_arm_group, "create_planner",
            lambda config=None, **kw: FakePlanner(FakeCollision()),
        )
        with pytest.raises(ValueError, match="not found in this group"):
            two_arm_group.check_collisions(arm_name="center")

    def test_no_contacts_returns_empty_list(self, two_arm_group, monkeypatch):
        monkeypatch.setattr(
            two_arm_group, "create_planner",
            lambda config=None, **kw: FakePlanner(FakeCollision()),
        )
        result = two_arm_group.check_collisions()
        assert result == []


# =============================================================================
# retime 
# =============================================================================


class TestRetime:
    def test_retime_calls_trajectory_from_path_with_group_limits(self, two_arm_group, monkeypatch):
        captured = {}

        def fake_from_path(**kwargs):
            captured.update(kwargs)
            return "TRAJECTORY_SENTINEL"

        monkeypatch.setattr(
            "mj_manipulator.arm_group.Trajectory.from_path", staticmethod(fake_from_path)
        )
        path = [np.zeros(4), np.ones(4)]
        result = two_arm_group.retime(path)

        assert result == "TRAJECTORY_SENTINEL"
        assert captured["path"] is path
        assert captured["entity"] == two_arm_group.config.name
        assert captured["joint_names"] == two_arm_group.joint_names
        np.testing.assert_array_equal(captured["vel_limits"], two_arm_group.kinematic_limits.velocity)
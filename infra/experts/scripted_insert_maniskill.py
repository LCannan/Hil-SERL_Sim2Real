"""Scripted stand-in for a human operator on the ManiSkill peg-insertion task.

Plans with mplib through ManiSkill's ``PandaArmMotionPlanningSolver`` and
converts the resulting joint-space waypoints into the ``pd_joint_delta_pos``
actions this environment expects.  Measured success rate from reset: 7/8.

**Do not use the shipped solver's ``move_to_pose_with_screw``/``follow_path``.**
They assert ``control_mode in ["pd_joint_pos", "pd_joint_pos_vel"]``, step the
environment themselves, and emit absolute ``qpos`` -- all three are wrong here.
We call the underlying ``planner.planner.plan_screw`` for waypoints and convert:

    a[:7] = clip((waypoint_q - qpos[:7]) / 0.1, -1, 1)   # 0.1 = Panda delta bound
    a[7]  = gripper                                       # +1 open / -1 close

The waypoint sequence mirrors
``mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py``:
reach, grasp, settle the gripper, pre-insert, refine against the live peg pose,
then insert.  Stages are planned lazily, one at a time -- planning the
refinement iterations up front reads a stale peg pose and roughly halves the
success rate.

The full sequence takes roughly 65-100 control steps, against the task's
100-step episode limit.  A takeover therefore has to happen early to finish:
measured over twelve seeded takeover states at k in {5..30} random steps, 12/12
are rescued when the episode is long enough and only 4/12 within the shipped
limit, the misses all being timeouts at exactly ``100 - k`` steps rather than
misbehavior.  That is a property of the budget, not of the expert.
"""

from typing import Any, List, Optional, Tuple

import numpy as np

from .base import ExpertBase

# Panda `pd_joint_delta_pos` bound: a normalized action of 1.0 commands +0.1 rad.
_JOINT_DELTA_BOUND = 0.1
_ARM_DOF = 7
_OPEN, _CLOSE = 1.0, -1.0
_FINGER_LENGTH = 0.025

class ScriptedInsertManiSkillExpert(ExpertBase):
    """Plan a grasp-align-insert sequence and replay it as joint deltas.

    Re-entrant: the plan is recomputed from the *current* robot and peg poses
    whenever the expert takes over, so it can rescue a policy-induced state
    rather than only running from ``reset``.  If the peg is already grasped the
    replan resumes at pre-insert rather than at the reach, which would open the
    gripper and drop it -- see :meth:`on_takeover`.  A stage whose plan fails is
    skipped rather than raising -- an unreachable waypoint should degrade the
    demonstration, not crash the actor.
    """

    # Stages 0-3 are reach / grasp / settle / pre-insert; refinement follows.
    _REFINE_START = 4

    def __init__(
        self,
        env: Any,
        joint_vel_limits: float = 0.5,
        joint_acc_limits: float = 0.5,
        grasp_settle_steps: int = 8,
        refine_iters: int = 3,
        action_dim: int = 8,
        # Accepted and ignored: the factory seeds every expert uniformly, and
        # this planner is deterministic given the scene.
        seed: int = 0,
        **kwargs: Any,
    ):
        if kwargs:
            raise TypeError(
                f"Unexpected expert_kwargs for scripted_insert_maniskill: {sorted(kwargs)}"
            )
        if env is None:
            raise ValueError(
                "ScriptedInsertManiSkillExpert needs the unwrapped "
                "ManiSkillPegInsertGymEnv to read the peg and goal poses."
            )

        self.env = env
        self.joint_vel_limits = float(joint_vel_limits)
        self.joint_acc_limits = float(joint_acc_limits)
        self.grasp_settle_steps = int(grasp_settle_steps)
        self.refine_iters = int(refine_iters)
        self.action_dim = int(action_dim)

        # env._env is the CPUGymWrapper; .unwrapped is the ManiSkill BaseEnv
        # exposing .peg / .goal_pose / .peg_half_sizes / .agent.
        inner = getattr(env, "_env", None)
        if inner is None:
            raise ValueError(
                "ManiSkillPegInsertGymEnv was built with fake_env=True; the "
                "scripted expert needs a live simulator."
            )
        self._base = inner.unwrapped
        self._planner = None
        self._queue: List[Tuple[np.ndarray, float]] = []
        self._plan_stage = 0
        self._peg_init_pose = None

    # ---------------------------------------------------------------- planning

    def _make_planner(self):
        from mani_skill.examples.motionplanning.panda.motionplanner import (
            PandaArmMotionPlanningSolver,
        )

        return PandaArmMotionPlanningSolver(
            self._base,
            debug=False,
            vis=False,
            base_pose=self._base.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_vel_limits=self.joint_vel_limits,
            joint_acc_limits=self.joint_acc_limits,
        )

    def _qpos(self) -> np.ndarray:
        return self._base.agent.robot.get_qpos().cpu().numpy()[0]

    def _grasp_pose(self):
        import sapien
        from mani_skill.examples.motionplanning.base_motionplanner.utils import (
            compute_grasp_info_by_obb,
            get_actor_obb,
        )

        obb = get_actor_obb(self._base.peg)
        approaching = np.array([0.0, 0.0, -1.0])
        closing_dir = (
            self._base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        )
        info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=closing_dir,
            depth=_FINGER_LENGTH,
        )
        pose = self._base.agent.build_grasp_pose(
            approaching, info["closing"], info["center"]
        )
        back_off = max(0.05, self._base.peg_half_sizes[0, 0].item() / 2 + 0.01)
        return pose * sapien.Pose([-back_off, 0.0, 0.0])

    def _plan_to(self, pose, gripper: float) -> List[Tuple[np.ndarray, float]]:
        """Waypoints toward ``pose``; empty when the plan fails.

        ``plan_screw`` produces a straight-line Cartesian motion, which is what
        the reference solution uses and is far cheaper.  It has no way around an
        obstacle, though, so it fails outright from a badly displaced pose --
        exactly the situation a HIL takeover lands in after the policy has
        wandered.  Fall back to sampling-based RRT there, as ManiSkill's own
        solver does.
        """
        goal = np.concatenate([pose.p.reshape(-1), pose.q.reshape(-1)])
        for plan in (self._planner.planner.plan_screw,
                     self._planner.planner.plan_qpos_to_pose):
            try:
                result = plan(goal, self._qpos())
            except Exception:
                continue
            if isinstance(result, dict) and "position" in result:
                return [(np.asarray(q), gripper) for q in result["position"]]
        return []

    def _refill(self) -> None:
        """Queue the next stage of the sequence, advancing until something plans.

        Stages are planned lazily, one at a time, and only once the previous
        stage's waypoints have all been consumed.  That laziness is load-bearing
        for the refinement stages: each one measures where the peg actually
        ended up after the last correction moved it.  Planning them in a batch
        makes every iteration read the same stale pose and compound the same
        correction, which cost roughly half the success rate.
        """
        import sapien

        last_stage = self._REFINE_START + self.refine_iters
        while not self._queue and self._plan_stage <= last_stage:
            stage, self._plan_stage = self._plan_stage, self._plan_stage + 1

            if stage == 0:
                self._planner = self._make_planner()
                self._peg_init_pose = self._base.peg.pose
                self._grasp = self._grasp_pose()
                self._queue = self._plan_to(
                    self._grasp * sapien.Pose([0.0, 0.0, -0.05]), _OPEN
                )
            elif stage == 1:
                self._queue = self._plan_to(self._grasp, _OPEN)
            elif stage == 2:
                # Hold still with the gripper closing so the grasp settles.
                hold = self._qpos()[:_ARM_DOF]
                self._queue = [(hold.copy(), _CLOSE)] * self.grasp_settle_steps
            elif stage == 3:
                self._insert_pose = (
                    self._base.goal_pose * self._peg_init_pose.inv() * self._grasp
                )
                self._offset = sapien.Pose(
                    [-0.01 - self._base.peg_half_sizes[0, 0].item(), 0.0, 0.0]
                )
                self._pre_insert = self._insert_pose * self._offset
                self._queue = self._plan_to(self._pre_insert, _CLOSE)
            elif stage < last_stage:
                # One refinement iteration against the live peg pose.
                delta = self._base.goal_pose * self._offset * self._base.peg.pose.inv()
                self._pre_insert = delta * self._pre_insert
                self._queue = self._plan_to(self._pre_insert, _CLOSE)
            else:
                self._queue = self._plan_to(
                    self._insert_pose * sapien.Pose([0.05, 0.0, 0.0]), _CLOSE
                )

    # --------------------------------------------------------------- interface

    def reset(self) -> None:
        self._queue = []
        self._plan_stage = 0
        self._peg_init_pose = None

    def on_takeover(self) -> None:
        """Replan from the current state when control comes back from the policy.

        Restarting unconditionally at stage 0 is wrong once the peg is in hand:
        that stage plans a *reach*, which commands an open gripper, so the peg
        is dropped within one step.  Under ``trigger=disagreement`` a takeover
        lasts only ``min_takeover_steps``, which is long enough to open the hand
        and not long enough to grasp again -- every correction would leave the
        episode worse than it found it.

        When the peg is already grasped, resume at the pre-insert stage instead
        and measure the grasp transform live, from where the gripper is holding
        the peg right now, rather than from the reach the expert never made.
        """
        self.reset()
        if not self._base.agent.is_grasping(self._base.peg):
            return

        self._planner = self._make_planner()
        # Stage 3 forms the insertion pose as `goal * peg_init.inv() * grasp`,
        # i.e. the gripper pose that carries the peg to the goal. Feeding it the
        # current peg and TCP poses expresses exactly that for the grasp in
        # hand, so the stage needs no special case.
        self._peg_init_pose = self._base.peg.pose
        self._grasp = self._base.agent.tcp.pose
        self._plan_stage = 3

    def get_action(self) -> Tuple[np.ndarray, List[int]]:
        self._refill()

        action = np.zeros(self.action_dim, dtype=np.float32)
        if not self._queue:
            # Sequence exhausted (or unplannable): hold position, stay closed.
            action[-1] = _CLOSE
            return action, [0, 0]

        waypoint, gripper = self._queue.pop(0)
        current = self._qpos()[:_ARM_DOF]
        action[:_ARM_DOF] = np.clip(
            (waypoint[:_ARM_DOF] - current) / _JOINT_DELTA_BOUND, -1.0, 1.0
        )
        action[-1] = gripper
        return action, [0, 0]

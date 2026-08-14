import numpy as np
import gymnasium as gym

# The one definition of the gripper encoding, shared with the environments and
# the grasp critic.  `robosuite_pick_place_gym_env` imports it the same way;
# a second copy of the thresholds here would mislabel training data rather than
# raise if the two ever drifted apart.
from algorithm.utils.gripper import CLOSE_THRESHOLD, OPEN_THRESHOLD


def _gripper_latch(command: float, current: bool) -> bool:
    """The environments' latch rule: only a decisive command changes it."""
    if command < CLOSE_THRESHOLD:
        return True
    if command > OPEN_THRESHOLD:
        return False
    return current


def SpaceMouseExpert(*args, **kwargs):
    """Deferred import of the HID driver.

    Kept as a module-level name so the two legacy wrappers below are unchanged,
    but resolved on first use: importing this module must not require HID
    libraries on a machine that only runs scripted experts.
    """
    from infra.hardware.spacemouse.spacemouse_expert import (
        SpaceMouseExpert as _SpaceMouseExpert,
    )

    return _SpaceMouseExpert(*args, **kwargs)


class ExpertIntervention(gym.Wrapper):
    """Human-in-the-loop intervention driven by a pluggable expert.

    Produces the same ``info["intervene_action"]`` contract that
    ``train_serl.py`` and ``record_demo.py`` consume, but takes its corrective
    action from any :class:`~infra.experts.base.Expert` -- a human on a
    SpaceMouse, or a scripted controller standing in for one.

    Two differences from :class:`SpacemouseIntervention`, which stays as-is
    because ``insert_real`` depends on it:

    - It is a ``gym.Wrapper``, not a ``gym.ActionWrapper``.  An ActionWrapper
      never sees the observation, and a scripted expert has to be told when an
      episode begins.
    - It never appends a gripper dimension.  The expert returns an action of the
      environment's full width, which is what makes this work for the 8-dim
      joint-space ManiSkill task as well as the 6-dim Cartesian ones.

    Like the original, an intervention *replaces* the policy action outright --
    HIL-SERL does not blend.

    Trigger modes:

    ``always``
        The expert drives the whole episode.  Used by ``record_demo.py`` to
        bootstrap demonstrations without any hardware.
    ``manual``
        Intervene while the expert's action is non-zero, i.e. whenever the human
        is actually pushing the device.  This is the original semantics.
    ``disagreement``
        Intervene when the policy and the expert disagree by more than
        ``disagreement_threshold``.  Approximates a human who steps in on seeing
        the robot about to go wrong.

    ``disagreement`` alone would either take over constantly (degenerating into
    behavior cloning on scripted data) or never.  Two limits keep it
    human-shaped: a minimum dwell time, because nobody lets go after one step,
    and an intervention budget that decays to zero over training so the policy
    is progressively left to fail on its own.

    An expert may also drive two things at runtime, both duck-typed so the
    scripted experts are untouched:

    - ``mode_toggle_requested`` flips between ``always`` and ``manual`` mid-run,
      which is how a SpaceMouse's right button switches between driving the
      whole episode and only correcting it.
    - ``sync_gripper`` receives the latch this wrapper computed from the action
      the environment actually executed.  See :meth:`_hold_gripper`.
    """

    _TRIGGERS = ("always", "manual", "disagreement")
    _FRAMES = ("base", "tcp")
    # `manual` toggles against this; `disagreement` is not in the cycle because
    # its budget bookkeeping keeps running while another mode is engaged, so
    # returning to it mid-run would find the budget already spent.
    _MODE_CYCLE = {"always": "manual", "manual": "always"}

    def __init__(
        self,
        env,
        expert,
        trigger: str = "disagreement",
        disagreement_threshold: float = 0.6,
        min_takeover_steps: int = 5,
        max_intervention_ratio: float = 0.4,
        intervention_decay_steps: int = 0,
        manual_deadband: float = 1e-3,
        expert_frame: str = "base",
        expert_frame_yaw: float = 0.0,
    ):
        super().__init__(env)
        if trigger not in self._TRIGGERS:
            raise ValueError(
                f"Unknown hil.trigger {trigger!r}. Expected one of {self._TRIGGERS}."
            )
        if expert_frame not in self._FRAMES:
            raise ValueError(
                f"Unknown hil.expert_frame {expert_frame!r}. "
                f"Expected one of {self._FRAMES}."
            )

        self.expert = expert
        self.trigger = trigger
        self.disagreement_threshold = float(disagreement_threshold)
        self.min_takeover_steps = int(min_takeover_steps)
        self.max_intervention_ratio = float(max_intervention_ratio)
        self.intervention_decay_steps = int(intervention_decay_steps)
        self.manual_deadband = float(manual_deadband)
        # `tcp` rotates the expert's translation and rotation triplets out of the
        # end-effector frame before they reach the environment, which is what
        # makes a hand-held device feel attached to the gripper: push away from
        # yourself and the tool goes where it is pointing, whatever the wrist has
        # rotated to.  `base` is the historical behaviour and the right choice
        # for the scripted experts, which compute their deltas from world-frame
        # geometry and would be rotated twice by this.
        self.expert_frame = expert_frame
        # Yaw offset, in degrees, applied about the tool's own z before the
        # wrist rotation.  A gripper's body axes are set by how the flange was
        # assembled, not by anything the operator can see: the Panda's tool x
        # points along world +y at this task's reset pose, so a raw `tcp` mapping
        # sends a forward push sideways on screen and a sideways push forward --
        # a 90-degree swap rather than a sign error, which is the giveaway.
        # -90 squares the device with the tool here; re-measure for another arm.
        self.expert_frame_yaw = float(expert_frame_yaw)

        self.left, self.right = False, False
        # The configured trigger stays put; this is the one in force, which the
        # operator may switch at runtime.  Kept apart so a config value is still
        # readable when diagnosing what a session actually did.
        self._active_trigger = trigger
        self._dwell = 0
        self._intervening = False
        self._total_steps = 0
        self._intervened_steps = 0
        # Held so `tcp` framing can read the wrist orientation the operator is
        # actually looking at when they push.
        self._last_tcp_pose = None
        self._last_expert = None
        self._last_executed = None
        # The gripper latch the environment is holding, tracked from the actions
        # actually executed.  Both gripper environments reset it to open.
        self._gripper_closed = False
        self._has_gripper = bool(self.action_space.shape[0] > 6)

    @property
    def _budget(self) -> float:
        """Allowed intervention fraction, decaying linearly to zero."""
        if self.max_intervention_ratio >= 1.0:
            return 1.0
        if self.intervention_decay_steps <= 0:
            return self.max_intervention_ratio
        progress = min(1.0, self._total_steps / self.intervention_decay_steps)
        return self.max_intervention_ratio * (1.0 - progress)

    def _should_intervene(self, policy_action, expert_action) -> bool:
        if self._active_trigger == "always":
            return True
        if self._active_trigger == "manual":
            # The Cartesian channels only.  The gripper is a latch the operator
            # holds asserted for the whole carry, so counting it here would put
            # the norm permanently above any deadband and the human would never
            # hand motion control back to the policy.
            return float(np.linalg.norm(expert_action[:6])) > self.manual_deadband

        # disagreement
        budget = self._budget
        if budget <= 0.0:
            return False
        # Finish what was started before re-checking the budget: releasing
        # control mid-correction leaves the robot in a worse state than never
        # having taken over.
        if self._dwell > 0:
            return True
        if self._total_steps > 0 and (self._intervened_steps / self._total_steps) >= budget:
            return False
        gap = float(np.linalg.norm(np.asarray(policy_action) - expert_action))
        return gap > self.disagreement_threshold

    def _remember_pose(self, obs) -> None:
        """Snapshot the base-frame `tcp_pose` for the next step's framing.

        A copy, not a reference: the wrappers outside this one mutate the very
        dict handed back here.  `RelativeFrame` rewrites `tcp_pose` in place to
        be relative to the reset pose and `Quat2RotvecWrapper` shortens it from
        7 entries to 6, so by the time the next `step` reads it, a stored
        reference would yield a 6-vector of near-zeros -- silently disabling the
        rotation instead of failing.
        """
        self._last_tcp_pose = None
        state = obs.get("state") if isinstance(obs, dict) else None
        if isinstance(state, dict) and "tcp_pose" in state:
            pose = np.asarray(state["tcp_pose"], dtype=np.float64).reshape(-1)
            if pose.shape[0] >= 7:
                self._last_tcp_pose = pose.copy()

    def _to_base_frame(self, action):
        """Rotate an end-effector-frame action into the base frame.

        Only the two leading triplets are touched, so a gripper channel rides
        through untouched -- the same contract `RelativeFrame` follows.  Returns
        the action unchanged when no usable `tcp_pose` was seen, which is what
        keeps this safe for the joint-space task.

        The rotation is flattened to yaw only: the operator's up/down stays the
        world's up/down.  Taking the wrist's full orientation would tie the lift
        axis to the tool's z, which points *downwards* whenever the gripper is
        aimed at the table -- so a push upwards would drive the arm into the
        bin.  Keeping gravity fixed while the horizontal axes follow the wrist
        is both what an operator expects and what most teleoperation rigs do.
        """
        if self.expert_frame != "tcp":
            return action
        if self._last_tcp_pose is None or action.shape[0] < 6:
            return action

        from scipy.spatial.transform import Rotation

        # Yaw of the tool about the world's z, plus the flange offset.  Built
        # from the tool x axis projected onto the horizontal plane rather than
        # from an euler decomposition, which is singular exactly where this task
        # spends its time -- pointing straight down.
        tool_x = Rotation.from_quat(self._last_tcp_pose[3:7]).as_matrix()[:, 0]
        yaw = np.arctan2(tool_x[1], tool_x[0]) + np.deg2rad(self.expert_frame_yaw)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rotation = np.array(
            [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]
        )

        action = action.copy()
        action[:3] = rotation @ action[:3]
        action[3:6] = rotation @ action[3:6]
        return action

    def _poll_mode_toggle(self) -> None:
        """Let the expert switch the trigger in force, if it asks to."""
        if not getattr(self.expert, "mode_toggle_requested", False):
            return
        target = self._MODE_CYCLE.get(self._active_trigger)
        if target is None:
            # Configured as `disagreement`: its budget keeps accruing while
            # another mode is engaged, so a round trip would return to a mode
            # whose budget is already spent.  Refuse rather than strand the run.
            print(
                f"[hil] mode switch ignored: trigger is {self._active_trigger!r}, "
                "which only `manual` and `always` toggle between."
            )
            return
        self._active_trigger = target
        # Printed, not merely shown on the HUD: `record_demo` runs with no
        # window on a headless host, and an accidental click that swapped the
        # mode there would silently write zero actions into the demo file.
        print(f"[hil] intervention mode -> {target}")

    def _hold_gripper(self, action):
        """Keep the gripper where the operator put it, whoever is steering.

        A click sets a latch the operator expects to survive the whole carry,
        but `manual` hands the rest of the action back to the policy the moment
        their hand leaves the puck -- and a policy with a grasp critic commands
        exactly -1/0/+1 on that channel every step, which clears the
        environment's +-0.5 threshold.  Without this the policy would drop
        whatever was being held, one step after it was grasped.

        Returns the action to *execute*, the action to *record*, and whether the
        channel was overridden.  Those first two differ on purpose, and the
        distinction is the whole point of this method:

        - The environment is an integrating actuator behind a latch, so the
          command has to be re-asserted every step or the grasp opens.
        - The recorded action feeds the discrete grasp critic, whose three
          classes are close/stay/open.  Re-asserting +-1 into the recording
          erases `stay` entirely: a SpaceMouse holds its latch at +-1 on every
          step (`gripper_scale: 1.0`), so a recorded demo came out 41.7% close /
          58.3% open / **0% stay** against the ~98.6% stay the class weighting
          expects.  `_grasp_class_weights` then reads a zero count for stay and
          weights it 1303x, teaching the head that holding never happens -- the
          measured `grasp_action_nonstay_frac == 1.000`, a gripper that flips
          open/closed every step and never carries anything.

        So: assert the latch to the env, record `stay` unless this is the step
        the operator actually flipped it.
        """
        if not self._has_gripper or self._last_expert is None:
            return action, action, False
        expert_gripper = float(self._last_expert[-1])
        if abs(expert_gripper) <= OPEN_THRESHOLD:
            return action, action, False  # expert is not asserting a latch

        executed = np.asarray(action, dtype=np.float32).copy()
        executed[-1] = expert_gripper

        # A flip is a change in the latch this command implies, measured against
        # the latch the environment is currently holding.  Only that step is a
        # close/open decision; the rest are `stay`.
        would_close = expert_gripper < CLOSE_THRESHOLD
        flipped = would_close != self._gripper_closed
        if flipped:
            recorded = executed
        else:
            recorded = np.asarray(action, dtype=np.float32).copy()
            recorded[-1] = 0.0
        return executed, recorded, True

    def reset(self, **kwargs):
        self._dwell = 0
        self._intervening = False
        self.left, self.right = False, False
        # Both gripper environments reset their latch to open; the expert is
        # told so its next click is not swallowed correcting a stale belief.
        self._gripper_closed = False
        sync = getattr(self.expert, "sync_gripper", None)
        if callable(sync):
            sync(False)
        self.expert.reset()
        obs, info = self.env.reset(**kwargs)
        self._remember_pose(obs)
        return obs, info

    def step(self, action):
        expert_action, buttons = self.expert.get_action()
        expert_action = np.asarray(expert_action, dtype=np.float32).reshape(-1)

        expected = self.action_space.shape[0]
        if expert_action.shape[0] != expected:
            raise ValueError(
                f"Expert returned a {expert_action.shape[0]}-dim action but the "
                f"environment's action space is {expected}-dim."
            )

        self.left = bool(buttons[0]) if len(buttons) > 0 else False
        self.right = bool(buttons[1]) if len(buttons) > 1 else False
        # Kept for `train/test_intervention.py`, which shows the operator what
        # their device emitted next to what the environment actually ran -- the
        # two differ under `expert_frame: tcp`.
        self._last_expert = expert_action
        self._poll_mode_toggle()

        # Triggering is judged on the expert's own output, before any frame
        # change: `manual`'s deadband is a property of how hard the operator is
        # pushing the device, and a rotation would leave its norm alone but make
        # the threshold read as if it applied to something else.
        intervened = self._should_intervene(action, expert_action)
        gripper_held = False
        if intervened:
            if not self._intervening:
                # A fresh takeover starts from whatever state the policy left
                # behind, so let the expert re-plan against it.  Keyed on
                # `_intervening`, not on the dwell counter: the counter hits
                # zero on every *continuing* takeover past min_takeover_steps,
                # and re-planning there would reset a scripted expert's state
                # machine every few steps and stall it forever.
                on_takeover = getattr(self.expert, "on_takeover", None)
                if callable(on_takeover):
                    on_takeover()
                self._dwell = self.min_takeover_steps
            self._dwell = max(0, self._dwell - 1)
            new_action = np.clip(
                self._to_base_frame(expert_action),
                self.action_space.low,
                self.action_space.high,
            )
            # Same close/stay/open bookkeeping as `_hold_gripper`, and needed
            # here for the same reason: `record_demo` drives with `trigger:
            # always`, so this is the branch that writes the demo files, and a
            # held latch recorded as +-1 every step is what erased `stay` from
            # them entirely.
            recorded_action = new_action
            if self._has_gripper:
                command = float(new_action[-1])
                if abs(command) > OPEN_THRESHOLD:
                    would_close = command < CLOSE_THRESHOLD
                    if would_close == self._gripper_closed:
                        recorded_action = np.asarray(
                            new_action, dtype=np.float32
                        ).copy()
                        recorded_action[-1] = 0.0
        else:
            self._dwell = 0
            new_action, recorded_action, gripper_held = self._hold_gripper(action)
        self._intervening = intervened
        # Also for test_intervention.py: what the environment was actually
        # handed, before RelativeFrame rewrites info["intervene_action"] into
        # the policy's frame on the way back out.
        self._last_executed = new_action if intervened else None

        self._total_steps += 1
        self._intervened_steps += int(intervened)

        obs, rew, done, truncated, info = self.env.step(new_action)
        self._remember_pose(obs)
        if intervened:
            # The rotated action, not the operator's raw one: this is what the
            # environment actually executed, and `RelativeFrame` outside will
            # map it into the policy's frame on the way out.  The gripper column
            # carries `stay` on the steps that merely hold the latch -- see
            # `_hold_gripper` for why recording +-1 there breaks the grasp head.
            info["intervene_action"] = recorded_action
        elif gripper_held:
            # Not an intervention -- the policy still steers -- but the action
            # stored for training has to be the one that ran, or the gripper
            # column teaches the critic the opposite of what happened.
            info["total_action"] = recorded_action
        if self._has_gripper:
            self._gripper_closed = _gripper_latch(
                float(new_action[-1]), self._gripper_closed
            )
            sync = getattr(self.expert, "sync_gripper", None)
            if callable(sync):
                sync(self._gripper_closed)
            info["gripper_closed"] = self._gripper_closed
        info["hil_mode"] = self._active_trigger
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def close(self):
        try:
            self.expert.close()
        finally:
            return self.env.close()


class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)
        intervened = False
        
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            return expert_a, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info


class DualSpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, gripper_enabled=True):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled

        self.expert = SpaceMouseExpert()
        self.left1, self.left2, self.right1, self.right2 = False, False, False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        intervened = False
        expert_a, buttons = self.expert.get_action()
        self.left1, self.left2, self.right1, self.right2 = tuple(buttons)


        if self.gripper_enabled:
            if self.left1:  # close gripper
                left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.left2:  # open gripper
                left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                left_gripper_action = np.zeros((1,))

            if self.right1:  # close gripper
                right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right2:  # open gripper
                right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                right_gripper_action = np.zeros((1,))
            expert_a = np.concatenate(
                (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
                axis=0,
            )

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
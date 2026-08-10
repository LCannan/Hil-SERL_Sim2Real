import numpy as np
import gymnasium as gym


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
    """

    _TRIGGERS = ("always", "manual", "disagreement")

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
    ):
        super().__init__(env)
        if trigger not in self._TRIGGERS:
            raise ValueError(
                f"Unknown hil.trigger {trigger!r}. Expected one of {self._TRIGGERS}."
            )

        self.expert = expert
        self.trigger = trigger
        self.disagreement_threshold = float(disagreement_threshold)
        self.min_takeover_steps = int(min_takeover_steps)
        self.max_intervention_ratio = float(max_intervention_ratio)
        self.intervention_decay_steps = int(intervention_decay_steps)
        self.manual_deadband = float(manual_deadband)

        self.left, self.right = False, False
        self._dwell = 0
        self._intervening = False
        self._total_steps = 0
        self._intervened_steps = 0

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
        if self.trigger == "always":
            return True
        if self.trigger == "manual":
            return float(np.linalg.norm(expert_action)) > self.manual_deadband

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

    def reset(self, **kwargs):
        self._dwell = 0
        self._intervening = False
        self.left, self.right = False, False
        self.expert.reset()
        return self.env.reset(**kwargs)

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

        intervened = self._should_intervene(action, expert_action)
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
                expert_action, self.action_space.low, self.action_space.high
            )
        else:
            self._dwell = 0
            new_action = action
        self._intervening = intervened

        self._total_steps += 1
        self._intervened_steps += int(intervened)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if intervened:
            info["intervene_action"] = new_action
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
"""Scripted stand-in for a human operator on the robosuite pick-and-place task.

Reads privileged simulator state (the object's pose) the same way a human reads
the screen, and drives the 7-DoF Cartesian action space.

Measure this expert **through the full wrapper stack**, with
``hil.trigger=always`` so it drives every step, exactly as ``record_demo`` does.
Driving a bare environment directly is misleading: ``RelativeFrame`` rotates
actions into the TCP frame, so an expert that scores 10/10 against the raw env
can fly the arm out of the workspace once wrapped.

The controller is a proportional servo on the end-effector position, sequenced
through seven stages.  Four details are load-bearing, and each one costs the
whole success rate on its own:

- **Grasp below the carton's top face, not at its centre.**  The carton's
  ``top_offset`` is well above its body origin (+0.059 m for the widened
  ``milk_wide.xml``, +0.075 m for robosuite's stock one), so a target at the
  centre puts the fingers well inside the box: they jam against the side and
  shove it across the bin instead of closing around it.  ``grasp_height`` is
  measured from the body origin and is **specific to the object's geometry** --
  swapping ``environment.config.object_xml`` means re-tuning it.  Measured
  through the full wrapper stack: 0.03 gives 6/6 on the wide carton, 0.045
  drops to 3/6, and the stock carton's 0.05 gives 2/6.
- **Every servo stage has a step budget as well as a tolerance.**  The gripper
  can come to rest against the carton with the remaining error physically
  unreachable, and a tolerance-only gate then waits there forever.  This was a
  0/10 -> 8/8 fix on its own.
- **RELEASE holds still for several steps.**  robosuite's gripper is rate-based
  -- it reads only ``sign(action)`` and integrates at ``speed=0.2`` -- so a
  single open command moves the fingers by a fifth of their travel and drops
  nothing.  The env's latch keeps the command asserted, but the fingers still
  need the steps to physically clear the carton.
- **RETREAT climbs away from the object afterwards.**  ``_check_success``
  requires ``r_reach = 1 - tanh(10*d) < 0.6``, i.e. the gripper more than
  **4.24 cm** from the object.  Placing it perfectly and hovering scores zero.

The phase is recomputed from current geometry on every call, so the expert is
re-entrant and can rescue a policy-induced state rather than only running from
``reset``.
"""

from typing import Any, List, Tuple

import numpy as np

from .base import ExpertBase

_APPROACH = "approach"
_DESCEND = "descend"
_GRASP = "grasp"
_LIFT = "lift"
_CARRY = "carry"
_RELEASE = "release"
_RETREAT = "retreat"

# This repo's gripper convention, which is the reverse of robosuite's; the
# environment translates.  See its module docstring.
_CLOSE, _OPEN = -1.0, 1.0


class ScriptedPickPlaceMilkExpert(ExpertBase):
    """Approach, grasp, carry to the target bin, release, and retreat."""

    def __init__(
        self,
        env: Any,
        gain: float = 1.0,
        hover_height: float = 0.12,
        grasp_height: float = 0.02,
        lift_height: float = 0.25,
        carry_height: float = 1.13,
        retreat_height: float = 1.30,
        approach_tol: float = 0.006,
        descend_tol: float = 0.008,
        lift_tol: float = 0.02,
        carry_tol: float = 0.03,
        dwell_steps: int = 15,
        phase_step_limit: int = 60,
        carry_step_limit: int = 100,
        action_dim: int = 7,
        # Accepted and ignored: the factory seeds every expert uniformly, and
        # this controller is deterministic given the scene.
        seed: int = 0,
        **kwargs: Any,
    ):
        # The workspace injects `display` whenever hil.hud is on; a scripted
        # expert has no use for it but must not reject it.
        kwargs.pop("display", None)
        if kwargs:
            raise TypeError(
                f"Unexpected expert_kwargs for scripted_pick_place_milk: {sorted(kwargs)}"
            )
        if env is None:
            raise ValueError(
                "ScriptedPickPlaceMilkExpert needs the unwrapped "
                "RobosuitePickPlaceGymEnv to read the object and goal poses."
            )
        if getattr(env, "_env", None) is None:
            raise ValueError(
                "RobosuitePickPlaceGymEnv was built with fake_env=True; the "
                "scripted expert needs a live simulator."
            )

        self.env = env
        self.gain = float(gain)
        self.hover_height = float(hover_height)
        self.grasp_height = float(grasp_height)
        self.lift_height = float(lift_height)
        self.carry_height = float(carry_height)
        self.retreat_height = float(retreat_height)
        self.approach_tol = float(approach_tol)
        self.descend_tol = float(descend_tol)
        self.lift_tol = float(lift_tol)
        self.carry_tol = float(carry_tol)
        self.dwell_steps = int(dwell_steps)
        self.phase_step_limit = int(phase_step_limit)
        self.carry_step_limit = int(carry_step_limit)
        self.action_dim = int(action_dim)

        self.reset()

    def reset(self) -> None:
        self._phase = _APPROACH
        self._dwell = 0
        self._phase_steps = 0

    def _advance(self, phase: str, dwell: int = 0) -> None:
        self._phase = phase
        self._dwell = dwell
        self._phase_steps = 0

    def on_takeover(self) -> None:
        """Resume from the current state when control comes back from the policy.

        Restarting unconditionally at ``_APPROACH`` is wrong once the carton is
        in hand: that stage commands an open gripper, so it is dropped within a
        few steps.  Under ``trigger=disagreement`` a takeover lasts only
        ``min_takeover_steps`` -- long enough to open the hand, not long enough
        to grasp again -- so every correction would leave the episode worse than
        it found it.

        When the object is already grasped, resume at the carry instead.  The
        remaining stages are all driven from live geometry, so nothing else
        needs restoring.
        """
        self.reset()
        if self.env._is_grasped():
            self._advance(_CARRY)

    def _eef_pos(self) -> np.ndarray:
        return np.asarray(
            self.env._env._get_observations()["robot0_eef_pos"], dtype=np.float64
        )

    def _object_pos(self) -> np.ndarray:
        obs = self.env._env._get_observations()
        return np.asarray(obs[f"{self.env._object_name}_pos"], dtype=np.float64)

    def get_action(self) -> Tuple[np.ndarray, List[int]]:
        eef = self._eef_pos()
        obj = self._object_pos()
        goal = self.env.goal_position
        grasped = self.env._is_grasped()
        self._phase_steps += 1

        # Each servo stage gets a step budget as well as a tolerance.  The
        # budget is what makes the sequence robust rather than merely tidy: the
        # gripper can end up resting against the carton with the remaining error
        # unreachable, and a tolerance-only gate then waits there forever while
        # pushing the object across the bin.  Timing out into the next stage is
        # what the measured 8/8 run did.
        timed_out = self._phase_steps >= (
            self.carry_step_limit if self._phase == _CARRY else self.phase_step_limit
        )

        gripper = _OPEN
        if self._phase == _APPROACH:
            target = obj + np.array([0.0, 0.0, self.hover_height])
            if np.linalg.norm(target - eef) < self.approach_tol or timed_out:
                self._advance(_DESCEND)
        elif self._phase == _DESCEND:
            # Relative to the carton's centre.  Its top face sits 7.5 cm above
            # that, so this grasps 2.5 cm below the top -- aiming at the centre
            # drives the fingers into the side of the box and shoves it away.
            target = obj + np.array([0.0, 0.0, self.grasp_height])
            if np.linalg.norm(target - eef) < self.descend_tol or timed_out:
                self._advance(_GRASP, self.dwell_steps)
        elif self._phase == _GRASP:
            # Hold still while the rate-based gripper closes.
            target = eef
            gripper = _CLOSE
            self._dwell -= 1
            if self._dwell <= 0:
                self._advance(_LIFT)
        elif self._phase == _LIFT:
            target = obj + np.array([0.0, 0.0, self.lift_height])
            gripper = _CLOSE
            if np.linalg.norm(target - eef) < self.lift_tol or timed_out:
                self._advance(_CARRY)
        elif self._phase == _CARRY:
            target = np.array([goal[0], goal[1], self.carry_height])
            gripper = _CLOSE
            if np.linalg.norm(target - eef) < self.carry_tol or timed_out:
                self._advance(_RELEASE, self.dwell_steps)
        elif self._phase == _RELEASE:
            target = eef
            self._dwell -= 1
            if self._dwell <= 0:
                self._advance(_RETREAT)
        else:  # _RETREAT -- climb clear so `r_reach` falls below 0.6
            target = np.array([goal[0], goal[1], self.retreat_height])

        # A grasp lost mid-carry makes the rest of the plan meaningless: go back
        # and pick the carton up again rather than placing an empty hand.
        if self._phase in (_LIFT, _CARRY) and not grasped:
            self._advance(_APPROACH)

        delta = target - eef
        action = np.zeros(self.action_dim, dtype=np.float32)
        # Dividing by the controller's own output scale makes a unit action
        # command exactly the remaining error, so retuning that scale does not
        # silently change the servo.  `gain` is a dimensionless multiplier on
        # top of it; 1.0 is the ratio the measured 8/8 run used.
        action[:3] = np.clip(
            delta / self.env.position_action_scale * self.gain, -1.0, 1.0
        )
        action[6] = gripper
        return action, [0, 0]

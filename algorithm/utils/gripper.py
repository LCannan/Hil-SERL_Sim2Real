"""The repo-wide gripper action encoding, shared by the envs and the grasp DQN.

`action[6]` is **latched, not integrated**: `< -0.5` closes, `> 0.5` opens, and
anything between holds.  A grasp has to survive a long carry during which the
operator pushes only translation and the gripper channel reads ~0; an
integrating channel would drift open exactly then.

The discrete grasp critic needs the same convention in *index* form, and the
two must not drift: a threshold mismatch would silently mislabel the training
data rather than raise.  Hence one definition here.

Deliberately dependency-free (torch + numpy only).  `algorithm/` and `infra/`
import nothing from each other -- the learner builds its env with
`fake_env=True` precisely so it never loads a simulator -- so this cannot live
in an env module.  The envs keep their own inline threshold checks and refer
back here; see `robosuite_pick_place_gym_env.step`.

Index order is ascending in the action value, so `index - 1` recovers the
command and the mapping matches upstream HIL-SERL's `round(a) + 1`:

    0 = close (-1.0),  1 = stay (0.0),  2 = open (+1.0)
"""

import numpy as np
import torch

CLOSE_THRESHOLD = -0.5
OPEN_THRESHOLD = 0.5

# Number of discrete gripper actions -- the paper's |A_2| for a single gripper
# ("open", "close", "stay", Sec. 3.3).
NUM_GRIPPER_ACTIONS = 3

GRIPPER_CLOSE_INDEX = 0
GRIPPER_STAY_INDEX = 1
GRIPPER_OPEN_INDEX = 2

# What each index commands.  Saturated rather than merely past the threshold, so
# a replayed action is unambiguous and matches what the SpaceMouse emits at
# `gripper_scale: 1.0`.
_INDEX_TO_ACTION = (-1.0, 0.0, 1.0)


def action_to_index(action):
    """Map a raw gripper command to its discrete class index.

    Accepts a torch Tensor or anything numpy can take, and returns the matching
    type (int64 tensor / array).  Thresholds rather than `round`: `round` only
    happens to agree when the expert's `gripper_scale` is exactly 1.0, and that
    is a per-task config value.
    """
    if isinstance(action, torch.Tensor):
        return torch.where(
            action < CLOSE_THRESHOLD,
            torch.full_like(action, GRIPPER_CLOSE_INDEX, dtype=torch.long),
            torch.where(
                action > OPEN_THRESHOLD,
                torch.full_like(action, GRIPPER_OPEN_INDEX, dtype=torch.long),
                torch.full_like(action, GRIPPER_STAY_INDEX, dtype=torch.long),
            ),
        )
    action = np.asarray(action)
    return np.where(
        action < CLOSE_THRESHOLD,
        GRIPPER_CLOSE_INDEX,
        np.where(action > OPEN_THRESHOLD, GRIPPER_OPEN_INDEX, GRIPPER_STAY_INDEX),
    ).astype(np.int64)


def index_to_action(index):
    """Inverse of :func:`action_to_index`, saturating to -1 / 0 / +1."""
    if isinstance(index, torch.Tensor):
        lookup = torch.tensor(
            _INDEX_TO_ACTION, dtype=torch.float32, device=index.device
        )
        return lookup[index]
    return np.asarray(_INDEX_TO_ACTION, dtype=np.float32)[np.asarray(index)]

"""Expert factory for human-in-the-loop training.

Every concrete expert is imported lazily so that ``import infra.experts`` never
pulls in ``pyspacemouse`` (which raises on a machine with no HID device) or
``mani_skill`` (an optional extra).
"""

from typing import Any

from .base import Expert, ExpertBase

__all__ = ["Expert", "ExpertBase", "make_expert", "available_experts"]

_EXPERTS = {
    "spacemouse": (".spacemouse", "SpaceMouseAdapter"),
    "keyboard": (".keyboard", "KeyboardExpert"),
    "scripted_insert_sim": (".scripted_insert_sim", "ScriptedInsertSimExpert"),
    "scripted_insert_maniskill": (
        ".scripted_insert_maniskill",
        "ScriptedInsertManiSkillExpert",
    ),
}


def available_experts() -> tuple:
    return tuple(sorted(_EXPERTS))


def make_expert(name: str, *, env: Any = None, **kwargs: Any) -> Expert:
    """Build an expert by name.

    ``env`` must be the *unwrapped* environment: the scripted experts read
    privileged simulator state from it.  Extra keyword arguments come from the
    task config's ``hil.expert_kwargs`` block.
    """
    import importlib

    if name not in _EXPERTS:
        choices = ", ".join(available_experts())
        raise ValueError(f"Unknown expert {name!r}. Available experts: {choices}")

    module_name, class_name = _EXPERTS[name]
    module = importlib.import_module(module_name, __name__)
    expert_cls = getattr(module, class_name)
    return expert_cls(env=env, **kwargs)

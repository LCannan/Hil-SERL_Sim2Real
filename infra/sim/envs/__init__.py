import ctypes.util
import importlib
import os


# MuJoCo otherwise falls back to GLFW, which requires an X display.  Keep an
# explicitly selected backend untouched, and on headless machines pick one that
# is actually installed: hardcoding a backend whose library is missing fails at
# `import mujoco` with an opaque OpenGL error.  EGL is preferred where present
# because it renders on the GPU; OSMesa is the software fallback.  Note that
# mani_skill imports mujoco transitively (via pytorch_kinematics), so this also
# governs the ManiSkill tasks.
if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
    for _backend, _library in (("egl", "EGL"), ("osmesa", "OSMesa")):
        if ctypes.util.find_library(_library):
            os.environ["MUJOCO_GL"] = _backend
            break

# Exports are resolved lazily (PEP 562) so that importing this package pulls in
# only the simulator the caller actually asked for.  In particular, a host that
# skipped the optional `maniskill` extra can still use the MuJoCo envs, and the
# learner -- which builds envs with fake_env=True -- never loads mujoco or
# sapien at all.
_LAZY_EXPORTS = {
    "PandaPegInsertGymEnv": "infra.sim.envs.panda_insert_gym_env",
    "PandaPegInsertDepthGymEnv": "infra.sim.envs.panda_insert_pointcloud_gym_env",
    "PandaPickCubeGymEnv": "infra.sim.envs.panda_pick_cube_gym_env",
    "ManiSkillPegInsertGymEnv": "infra.sim.envs.maniskill_peg_gym_env",
    "RobosuitePickPlaceGymEnv": "infra.sim.envs.robosuite_pick_place_gym_env",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # Import once; subsequent lookups skip __getattr__.
    return value


def __dir__():
    return sorted([*globals(), *_LAZY_EXPORTS])

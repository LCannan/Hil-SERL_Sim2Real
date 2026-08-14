"""Interactive check that a DeviceSpec's axis mapping matches the hardware.

Written for the SpaceMouse Wireless BT (0x256F:0xC63A), whose spec was cloned
from the cabled model -- Bluetooth-Edition firmware need not lay its HID report
out the same way, so the mapping has to be confirmed against the device rather
than assumed.

Reports what the *driver* decodes, not raw bytes: that is the layer the spec
governs, and the layer a wrong ``scale`` or ``byte1`` shows up in.  The extra
sign flip in ``SpaceMouseExpert._read_spacemouse`` sits above this, so a
correct result here still leaves the robot-frame convention to verify.

    uv run python infra/hardware/spacemouse/calibrate_spacemouse.py
"""

import time

from infra.hardware.spacemouse import pyspacemouse

# (prompt, attribute, expected sign) -- the sign is this driver's own
# convention, taken from the cabled SpaceMouse Wireless spec.  Translations are
# phrased to be unambiguous; the rotation prompts name an explicit direction
# because "clockwise" depends on which side you view the puck from.
_STEPS = [
    ("水平推向前方（远离你），不要抬起或倾斜", "y", +1),
    ("水平拉向后方（朝向你）", "y", -1),
    ("水平推向右侧", "x", +1),
    ("水平推向左侧", "x", -1),
    ("竖直向上提起", "z", +1),
    ("竖直向下按压", "z", -1),
    ("俯视手柄，把帽子水平扭向右（顺时针），不要平移", "yaw", -1),
    ("把帽子前缘向下压、后缘抬起（点头）", "pitch", -1),
    ("把帽子右缘向下压、左缘抬起", "roll", +1),
]

_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
_DEADBAND = 0.15
_HOLD_SECONDS = 2.5


def _wait_for_centre(device, timeout=3.0):
    """Block until the puck reads as centred.

    ``DeviceSpec.read`` is a state query, not a queue pop: it returns the
    cached ``tuple_state`` even when no new HID data arrived, so the last
    non-zero reading of the *previous* gesture persists indefinitely and would
    be measured as the next one's motion.  Polling until it decays is the only
    way to know the spring-back has been observed.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = device.read()
        if state is not None and all(
            abs(float(getattr(state, axis, 0.0))) < _DEADBAND for axis in _AXES
        ):
            return True
        time.sleep(0.01)
    return False


def _gesture(device, seconds):
    """Return the single sample with the largest combined deflection.

    One snapshot rather than a per-axis maximum: independent per-axis peaks mix
    the gesture with whatever the puck did on its way back to centre, and give
    every axis its own instant, which is not a pose the device was ever in.
    """
    best = {axis: 0.0 for axis in _AXES}
    best_norm = 0.0
    deadline = time.time() + seconds
    while time.time() < deadline:
        state = device.read()
        if state is not None:
            sample = {axis: float(getattr(state, axis, 0.0)) for axis in _AXES}
            norm = sum(value * value for value in sample.values())
            if norm > best_norm:
                best_norm, best = norm, sample
        time.sleep(0.005)
    return best


def main():
    print("connected:", pyspacemouse.list_devices())
    device = pyspacemouse.open(set_nonblocking_loop=True)
    if device is None:
        raise SystemExit("打开设备失败。检查 /dev/hidraw* 权限。")

    print("\n每一步：按回车 -> 立刻做动作并保持 2.5 秒 -> 松手让手柄回中。\n")
    problems = []
    for prompt, axis, sign in _STEPS:
        input(f">>> {prompt}\n    松手让手柄回中，然后按回车")
        if not _wait_for_centre(device):
            print("    ! 手柄未回中（或仍在动），本步读数可能不准\n")
        peak = _gesture(device, _HOLD_SECONDS)

        moved = {a: v for a, v in peak.items() if abs(v) > _DEADBAND}
        detail = "  ".join(f"{a}={v:+.2f}" for a, v in sorted(
            moved.items(), key=lambda kv: -abs(kv[1]))) or "(无输出)"

        got = peak[axis]
        dominant = max(_AXES, key=lambda a: abs(peak[a]))
        if abs(got) <= _DEADBAND:
            verdict = f"✗ 期望 {axis} 变化，实际：{detail}"
        elif dominant != axis:
            verdict = f"✗ 主轴是 {dominant} 而非 {axis}：{detail}"
        elif (got > 0) != (sign > 0):
            verdict = f"✗ {axis} 符号相反（期望 {sign:+d}，实测 {got:+.2f}）"
        else:
            verdict = f"✓ {axis}={got:+.2f}   {detail}"
        if verdict.startswith("✗"):
            problems.append((prompt, axis, sign, peak))
        print(f"    {verdict}\n")

    # Not pyspacemouse.close(): the module-level helper assumes _active_device
    # is a single DeviceSpec, but open() leaves a list there when the unit
    # exposes several HID interfaces (this one exposes three).
    device.close()

    if not problems:
        print("全部通过：device_specs 里的 SpaceMouse Wireless BT 映射与硬件一致。")
    else:
        print(f"{len(problems)} 步不符，需要修 DeviceSpec。把下面输出发回给 Claude：")
        for prompt, axis, sign, peak in problems:
            values = " ".join(f"{a}={peak[a]:+.2f}" for a in _AXES)
            print(f"  [{prompt}] 期望 {axis}{sign:+d} | 实测 {values}")


if __name__ == "__main__":
    main()

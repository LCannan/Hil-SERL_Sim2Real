---
name: spacemouse-teleop
description: Drive, tune, or debug SpaceMouse teleoperation in this repo - the two intervention modes (continuous / on-demand), the gripper toggle, response gains (pos_scale, rot_scale, deadzone, expo), expert_frame, the HUD, and device problems (no device found, axes wrong, buttons dead, clicks missed). Use when recording demos by hand, tuning how the puck feels, or when something about the SpaceMouse behaves oddly.
---

# SpaceMouse teleoperation

The 3Dconnexion puck is one of five experts behind `ExpertIntervention`
(`infra/wrappers/intervention.py`). It supplies the corrective action in a
human-in-the-loop run, and drives `record_demo` outright.

Hardware here: **SpaceMouse Wireless BT**, `0x256F:0xC63A`. That product id is
reported on every connection path, cabled USB included.

## Controls

| input | does |
| --- | --- |
| puck (6-DoF) | translate + rotate the end effector |
| **left button** | **toggle** the gripper: click closes, click again opens |
| **right button** | **toggle** the intervention mode |

Both buttons are **edge-triggered toggles**, not hold-to-act. Holding one does
nothing extra; each press flips a state.

### The two intervention modes

Right button switches the trigger in force at runtime. The HUD shows which is
live, and every switch prints `[hil] intervention mode -> ...` to stdout.

| HUD | trigger | behaviour |
| --- | --- | --- |
| `MODE: ON-DEMAND` | `manual` | the policy drives; you take over only while pushing the puck, and control returns the moment your hand leaves it |
| `MODE: CONTINUOUS` | `always` | you drive every step, the policy's action is discarded |

Starts from the task's configured `hil.trigger` and **persists across
episodes** — `env.reset` does not put it back, because silently reclaiming
control at an episode boundary is what an operator cannot see coming.

A task configured `trigger: disagreement` (the scripted-expert `_hil` configs)
**refuses** the switch and says so: its intervention budget keeps accruing while
another mode is engaged, so a round trip would return to a mode whose budget is
already spent.

### The gripper is yours, in both modes

Once you click it closed, the latch is held on every step **even while the
policy is steering**. This is deliberate and not a bug: with `grasp_critic: true`
the policy commands exactly -1/0/+1 on the gripper channel every step, which
clears the env's ±0.5 latch threshold — a grasp handed back to the policy is
dropped one step later. Those steps report `info["total_action"]` rather than
`info["intervene_action"]`: the action executed, but not an intervention.

Consequences worth knowing:

- `manual`'s deadband measures `expert_action[:6]` only. Counting the held
  gripper would pin the norm above any deadband and you could never hand motion
  control back. (The **keyboard** expert still has that bug — on a 7-dim task
  one press of space holds control for the rest of the run.)
- The latch resets to open on `env.reset`, matching both gripper envs, and the
  wrapper re-syncs the expert so your first click of a new episode is not
  swallowed correcting a stale belief.

## Tuning the feel

`hil.expert_kwargs` in the task config. Shipped values:

| key | `pick_place_milk_human` | `pick_cube_sim_human` | what it does |
| --- | --- | --- | --- |
| `pos_scale` | 0.96 | 0.4 | output at full deflection, translation |
| `rot_scale` | 0.85 | 0.15 | same, rotation |
| `deadzone` | 0.2 | 0.2 | per-axis fraction ignored; kills hand crosstalk |
| `expo` | 1.5 | 1.5 | >1 flattens the centre for fine work |
| `gripper_scale` | 1.0 | 1.0 | 1.0 saturates the ±0.5 latch threshold |

Two things that are measured, not guessed, and will mislead you if assumed:

- **Gains are per-step displacement, so speed depends on the rate.** These
  values were sized against the wall clock at 20 Hz (0.25 → 0.063 m/s, 0.5 →
  0.130, 0.96 → 0.249). `pick_place_milk` has since dropped to
  `actor_rate_hz: 10` alongside `control_freq: 10`, so a step now covers twice
  the distance and the arm moves about twice as fast as those figures — re-time
  it before trusting a number here.
- **Rotation saturates and translation does not.** The OSC's
  achieved-to-commanded ratio is a flat 0.26 for translation, but 0.23 → 0.19 →
  0.14 for rotation as you push harder. Matching a target angular rate needs
  `rot_scale` 0.85, not a proportional scaling.

`environment.config.action_limit` clips the human as well as the policy, so it
can never sit **below** `pos_scale`/`rot_scale` — a cap under the gains silently
limits the operator and pushing harder does nothing. It is `null` on
`pick_place_milk` for that reason. **The remaining lever for speed is
`control_freq`** (with `training.actor_rate_hz` moved to match).

### `hil.expert_frame`

- `base` (default) — the raw device axes pass through. What the **scripted**
  experts need: they compute deltas from world-frame geometry.
- `tcp` — the two leading triplets rotate with the wrist, so the puck feels
  attached to the gripper. `pick_place_milk_human` uses this.

`tcp` is **yaw-only** on purpose. Using the wrist's full orientation ties the
lift axis to the tool's z, which points *down* when the gripper faces the table
— a push upwards would drive the arm into the bin. Up stays world-up.

`hil.expert_frame_yaw` (degrees) corrects the flange offset. The Panda's tool x
runs along world +y, so a raw mapping swaps forward with sideways; **a swap
rather than a mirror is the signature**, and `pick_place_milk_human` uses 90.
A mirror instead (forward↔backward *and* left↔right, up unchanged) means the
value is 180° out. Re-measure for another arm.

The trigger is judged on the expert's **raw** output, before this rotation:
`manual_deadband` is a property of how hard you are pushing.

## The HUD

`hil.hud: true` opens one cv2 window (skipped with no `DISPLAY`, because Qt
aborts the process inside `imshow` before any catchable exception).

- `hud_render_size: 640` re-renders your view at that resolution instead of
  upscaling the policy's 128×128. The blur is a *resolution* limit — at 128 the
  gripper's own fingers are a few pixels across. Also sets the window size
  (1280×766 at 640). Costs ~7 ms/step against the 100 ms a 10 Hz loop allows.
- `hud_cameras` **pins** which cameras you see, so adding one to
  `training.image_keys` changes what the network sees without rearranging your
  window.
- `hud_show_policy_view: true` draws the policy's own 128×128 frames beneath,
  captioned `POLICY INPUT`, deliberately not upscaled.

**You see more than the policy does.** Same cameras, 640 vs 128. Avoid
demonstrating judgements that depend on detail only the high-resolution view
carries — the network cannot learn from what it cannot resolve.

## Workflows

```bash
# Feel out the device; writes NOTHING.  Prints what the expert emitted next to
# what the env ran, plus the trigger and deadband at startup.
uv run python -m train.test_intervention --exp_name=pick_place_milk_human

# Record demos by hand.  Keep the mode ON-DEMAND (`manual`) so idle steps record
# as the zero actions that genuinely executed.
uv run python -m train.record_demo --exp_name=pick_place_milk_human --successes_needed=20

# Check what you recorded, then replay it.
uv run python -m train.verify_demo --demo_path=demo_data/<file>.pkl
```

`run_hil_serl.sh` takes `EXPERT=spacemouse`, `TRIGGER=`, `DEMO_TRIGGER=` as
env-var overrides.

`record_demo` drives the env with **zero** actions and records whatever the
expert overrides them with. A stray right click there flips you to CONTINUOUS
and back — under CONTINUOUS every step is recorded as yours, which is fine, but
under ON-DEMAND the idle steps record as zeros. Watch the printed mode line.

## Device problems

| symptom | cause |
| --- | --- |
| `ImportError` about HID libraries | install `libhidapi-hidraw0`; check `/dev/hidraw*` is readable by this user |
| `RuntimeError: Could not open a SpaceMouse` | not plugged in, or no permission on `/dev/hidraw*` |
| axes wrong / crosstalk | `uv run python infra/hardware/spacemouse/calibrate_spacemouse.py` — reports what the *driver* decodes, which is the layer `DeviceSpec` governs |
| device opens but nothing moves | `uv run python infra/hardware/spacemouse/test_spacemouse.py` — a manual probe, not a pytest test |
| forward push moves the arm sideways | `expert_frame_yaw` is 90° out |
| push right moves the arm left on screen | camera mirroring: `expert_frame: tcp`, or `invert_xy: true` in `expert_kwargs` |
| arm never gives control back to the policy | something keeps `‖action[:6]‖` above `manual_deadband` — a puck that has not re-centred; raise `deadzone` |

### Why button presses are counted, not sampled

`SpaceMouseExpert._read_spacemouse` maintains a per-button **press counter** in
the free-running reader process, and the adapter diffs it. A control step is
100 ms at 10 Hz and a crisp click can begin and end between two samples of the
button *level*. Under the old hold-to-close binding a missed sample only meant
"hold it a bit longer"; under a toggle a dropped edge leaves your next click
**inverted**. The counter also cannot be forged by the reader's exception
handler, which zero-fills `buttons` and would otherwise synthesize a
release/press pair — a phantom toggle mid-carry.

## Layout

| file | role |
| --- | --- |
| `infra/experts/spacemouse.py` | `SpaceMouseAdapter`: edge detection, response shaping, the two toggles |
| `infra/hardware/spacemouse/spacemouse_expert.py` | reader process, press counters |
| `infra/hardware/spacemouse/pyspacemouse.py` | HID driver; `DeviceSpec` for `0xC63A` |
| `infra/wrappers/intervention.py` | `ExpertIntervention`: mode switching, gripper hold |
| `infra/utils/teleop_display.py` | the cv2 window |

`SpacemouseIntervention` in the same wrapper module is the **legacy** path, kept
because `insert_real` depends on it. It keeps the old momentary semantics (hold
left to close, right to open) and does not go through `SpaceMouseAdapter`.
Do not confuse the two when changing button behaviour.

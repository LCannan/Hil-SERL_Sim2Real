import warnings
warnings.filterwarnings("ignore")

import logging
import glob
logging.getLogger('asyncio').setLevel(logging.ERROR)

import importlib.util
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
import tqdm
from absl import app, flags
import os
import copy
import threading
import queue
import re

import pickle as pkl
from gymnasium.wrappers import RecordEpisodeStatistics

import torch
from algorithm.utils.torch_utils import dict_apply
from algorithm.utils.timer_utils import Timer
from algorithm.utils.train_utils import (
    concat_batches, 
    state_dict_to_numpy, 
    numpy_to_state_dict, 
    print_green
)
from algorithm.launch import (
    make_trainer_config,
    make_wandb_logger,
)

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from algorithm.data.data_store import ReplayBufferDataStore
from infra.utils.rate_limit import RateLimiter, resolve_rate_hz
from workspace import SERLWorkspace


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_multi_string(
    "config_override",
    [],
    "Hydra task overrides, for example training.batch_size=128.",
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_float(
    "actor_rate_hz",
    0.0,
    "Cap the actor's rollout rate, in environment steps per wall-clock second. "
    "0 means unthrottled. Overrides training.actor_rate_hz. Matters for "
    "human-in-the-loop runs: unthrottled the sim actor outruns the learner "
    "roughly 11:1, so corrections take effect long after the operator made "
    "them.",
)
flags.DEFINE_integer(
    "pretrain_steps",
    0,
    "Behavior-cloning steps to run on the demo buffer before any online data "
    "arrives. An untrained policy flails, and shrinking its exploration noise "
    "does not help -- only a gradient on the mean layer does. Writes a "
    "checkpoint on completion so the actor starts from it.",
)
flags.DEFINE_boolean(
    "pretrain_only",
    False,
    "Exit after pretraining instead of waiting for an actor. Use this to "
    "produce a warm-start checkpoint, then launch learner and actor normally "
    "against the same --checkpoint_path.",
)
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_string(
    "eval_video_dir",
    None,
    "Directory for evaluation MP4 videos. Defaults to <checkpoint_path>/videos.",
)
flags.DEFINE_integer(
    "eval_video_fps",
    20,
    "Evaluation video playback FPS. Every control-step frame is preserved.",
)
flags.DEFINE_float(
    "eval_video_start_hold_seconds",
    0.5,
    "Seconds to hold the initial frame in each evaluation video.",
)
flags.DEFINE_float(
    "eval_video_end_hold_seconds",
    1.0,
    "Seconds to hold the terminal frame in each evaluation video.",
)
flags.DEFINE_string(
    "eval_video_main_camera",
    "front",
    "Main simulation camera to place above wrist views; empty disables it.",
)
flags.DEFINE_boolean("debug", False, "Debug mode.")  


def _grasp_critic_enabled(ws) -> bool:
    """Whether this task trains a discrete gripper head.

    Read off the `agent:` block with `.get`, not through `BaseWorkspace`'s
    attribute forwarding (which only reaches `config.training`), and defaulted so
    the eleven task configs without the key are untouched -- OmegaConf raises on
    a missing key even in a branch that is never taken.
    """
    agent_config = ws.raw_config.get("agent") or {}
    return bool(agent_config.get("grasp_critic", False))


class _BufferDumpWriter:
    """Pickle transition batches to disk off the actor's control thread.

    The dump used to run inline, and at 1000 transitions it costs a measured
    244 ms -- against the 100 ms a 10 Hz control step allows.  That lands as a
    frozen frame in an operator's hand, and `RateLimiter.sleep` does not absorb
    it: an overrun takes the `else` branch and drops the backlog, so the whole
    dump duration is lost from the episode.  A 5.7 s stall was observed this way.

    The queue is deliberately shallow and **drops rather than blocks** when the
    disk cannot keep up.  These dumps are a convenience for resuming a run; the
    operator's feel is not.  A drop says so on stdout rather than silently
    stealing time back.
    """

    def __init__(self, keep_last: int = 0, maxsize: int = 2):
        self._queue = queue.Queue(maxsize=maxsize)
        self._keep_last = int(keep_last)
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._run, name="buffer-dump", daemon=True
        )
        self._thread.start()

    def submit(self, directory: str, step: int, batch: list) -> None:
        """Hand a batch over, or drop it if the writer is still busy."""
        try:
            self._queue.put_nowait((directory, step, batch))
        except queue.Full:
            self._dropped += 1
            print_green(
                f"buffer dump at step {step} dropped -- writer still busy "
                f"({self._dropped} total). Raise training.buffer_period if this "
                "repeats; the run is unaffected apart from resume coverage."
            )

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            directory, step, batch = item
            try:
                os.makedirs(directory, exist_ok=True)
                path = os.path.join(directory, f"transitions_{step}.pkl")
                with open(path, "wb") as f:
                    pkl.dump(batch, f)
                self._prune(directory)
            except Exception as exc:  # a full disk must not kill the rollout
                print_green(f"buffer dump to {directory} failed: {exc!r}")
            finally:
                self._queue.task_done()

    def _prune(self, directory: str) -> None:
        """Keep only the newest `keep_last` dumps in this directory.

        Left unbounded these accumulate at ~11 GB/hour and are never read again
        except at learner startup, which re-reads *all* of them -- so an old
        directory slows every subsequent run and evicts the page cache the
        desktop is living in.
        """
        if self._keep_last <= 0:
            return
        files = sorted(
            glob.glob(os.path.join(directory, "transitions_*.pkl")),
            key=lambda p: int(re.search(r"transitions_(\d+)\.pkl$", p).group(1)),
        )
        for stale in files[: -self._keep_last]:
            try:
                os.remove(stale)
            except OSError:
                pass

    def close(self) -> None:
        """Flush everything still queued, then stop the thread."""
        self._queue.put(None)
        self._thread.join()


def _add_camera_label(frame, label):
    frame = np.ascontiguousarray(frame.copy())
    cv2.rectangle(frame, (0, 0), (92, 24), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        label,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return frame


def _video_frame_from_observation(obs, image_keys):
    """Build a video frame from the latest image of each observation camera."""
    frames = []
    for key in image_keys:
        if key not in obs:
            continue
        frame = np.asarray(obs[key])
        while frame.ndim > 3:
            frame = frame[-1]
        if frame.ndim == 2:
            finite = np.isfinite(frame)
            if finite.any():
                low, high = np.percentile(frame[finite], (1, 99))
                scale = max(float(high - low), 1e-6)
                frame = np.clip((frame - low) / scale * 255.0, 0, 255)
            else:
                frame = np.zeros_like(frame)
            frame = np.repeat(frame[..., None], 3, axis=-1)
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            continue
        frames.append(
            _add_camera_label(
                np.asarray(frame[..., :3], dtype=np.uint8),
                key,
            )
        )

    if not frames:
        raise ValueError(
            "Evaluation video requires at least one RGB or depth image in the "
            f"observation; configured image keys: {list(image_keys)}"
        )

    target_height = min(frame.shape[0] for frame in frames)
    frames = [frame[:target_height] for frame in frames]
    return np.concatenate(frames, axis=1)


def _evaluation_video_frame(env, obs):
    wrist_row = _video_frame_from_observation(obs, workspace.image_keys)
    camera_name = FLAGS.eval_video_main_camera
    render_camera = getattr(env.unwrapped, "render_camera", None)
    if not camera_name or render_camera is None:
        return wrist_row

    main_frame = np.asarray(
        render_camera(
            camera_name=camera_name,
            width=wrist_row.shape[1],
            height=192,
        ),
        dtype=np.uint8,
    )[..., :3]
    main_frame = _add_camera_label(main_frame, camera_name)
    if main_frame.shape[1] != wrist_row.shape[1]:
        raise ValueError("Main and wrist video rows must have the same width")
    return np.concatenate([main_frame, wrist_row], axis=0)


def _find_ffmpeg_executable():
    override = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if override and os.path.isfile(override):
        return override

    spec = importlib.util.find_spec("imageio_ffmpeg")
    if spec is not None and spec.origin is not None:
        binary_dir = Path(spec.origin).parent / "binaries"
        bundled = sorted(binary_dir.glob("ffmpeg-*"))
        if bundled:
            return str(bundled[0])

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        return system_ffmpeg
    raise RuntimeError(
        "FFmpeg was not found. Run `uv sync` to install imageio-ffmpeg."
    )


def _write_video(video_path, frames, fps):
    height, width = frames[0].shape[:2]
    if height % 2 or width % 2:
        raise ValueError("Evaluation videos require even width and height")

    extension = Path(video_path).suffix.lower()
    if extension == ".webm":
        output_options = [
            "-c:v",
            "libvpx",
            "-deadline",
            "good",
            "-cpu-used",
            "2",
            "-crf",
            "10",
            "-b:v",
            "1M",
            "-pix_fmt",
            "yuv420p",
        ]
    elif extension == ".mp4":
        output_options = [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-profile:v",
            "baseline",
            "-level:v",
            "3.0",
            "-tag:v",
            "avc1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    else:
        raise ValueError(
            f"Unsupported evaluation video extension {extension!r}; "
            "use .webm or .mp4"
        )

    command = [
        _find_ffmpeg_executable(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        *output_options,
        video_path,
    ]
    for frame in frames:
        if frame.shape[:2] != (height, width):
            raise ValueError("All evaluation video frames must have the same size")
    raw_video = b"".join(
        np.ascontiguousarray(frame[..., :3], dtype=np.uint8).tobytes()
        for frame in frames
    )
    result = subprocess.run(
        command,
        input=raw_video,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"FFmpeg failed to write {video_path} (exit {result.returncode}): {stderr}"
        )


def evaluate(agent, env, device: str, checkpoint_step: int):
    """Run deterministic evaluation episodes and save one video per episode."""
    agent.eval()
    video_dir = FLAGS.eval_video_dir
    if video_dir is None:
        video_dir = os.path.join(FLAGS.checkpoint_path, "videos")
    os.makedirs(video_dir, exist_ok=True)

    fps = FLAGS.eval_video_fps
    successes = 0
    returns = []
    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset(seed=FLAGS.seed + episode)
        episode_return = 0.0
        initial_frame = _evaluation_video_frame(env, obs)
        frames = [initial_frame] * max(
            1, int(round(fps * FLAGS.eval_video_start_hold_seconds))
        )
        trajectory_frame_count = 1
        terminated = truncated = False

        while not (terminated or truncated):
            with torch.no_grad():
                obs_tensor = dict_apply(
                    obs, lambda x: torch.as_tensor(x, device=device)
                )
                action = agent.sample_actions(
                    observations=obs_tensor,
                    argmax=True,
                ).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            frames.append(_evaluation_video_frame(env, obs))
            trajectory_frame_count += 1

        success = bool(info.get("succeed", terminated))
        successes += int(success)
        returns.append(episode_return)
        frames.extend(
            [frames[-1]]
            * int(round(fps * FLAGS.eval_video_end_hold_seconds))
        )
        video_path = os.path.join(
            video_dir,
            f"checkpoint_{checkpoint_step}_episode_{episode:03d}_"
            f"{'success' if success else 'failure'}.mp4",
        )
        _write_video(video_path, frames, fps)
        print_green(
            f"Evaluation episode {episode + 1}/{FLAGS.eval_n_trajs}: "
            f"return={episode_return:.3f}, success={success}, "
            f"trajectory_frames={trajectory_frame_count}, "
            f"video_duration={len(frames) / fps:.2f}s, video={video_path}"
        )

    print_green(
        f"Evaluation finished: success_rate={successes / FLAGS.eval_n_trajs:.3f}, "
        f"mean_return={np.mean(returns):.3f}, videos={video_dir}"
    )


def actor(
    agent, 
    data_store, 
    intvn_data_store, 
    env,
    device: str = "cuda"
):
    agent.eval()
    parameter_lock = threading.Lock()
    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }
    
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
    )

    def update_params(params):
        """Update agent parameters from server"""
        state_dict = numpy_to_state_dict(params, device)
        with parameter_lock:
            agent.load_state_dict(state_dict, strict=False)

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []
    # These two lists exist only to be pickled into `checkpoint_path/buffer` every
    # `buffer_period` steps.  When either is off, nothing ever clears them, and at
    # 192 KiB per deepcopied transition that grew at ~13.5 GiB/hour until the
    # machine died -- so decide once, here, rather than accumulating for a flush
    # that will not come.
    dump_transitions = bool(FLAGS.checkpoint_path) and workspace.buffer_period > 0
    if not dump_transitions:
        print_green(
            "buffer dumps disabled (no --checkpoint_path or buffer_period=0): "
            "transitions will not be retained in actor memory"
        )
    # Read with getattr so the tasks that predate the key keep the old
    # keep-everything behaviour rather than silently starting to delete.
    dump_writer = (
        _BufferDumpWriter(keep_last=getattr(workspace, "buffer_keep_last", 0))
        if dump_transitions
        else None
    )

    obs, _ = env.reset()
    timer = Timer()
    running_return = 0.0
    intervention_count = 0
    intervention_steps = 0
    already_intervened = False

    # Pace the rollout against the wall clock.  Unthrottled, this loop runs at
    # ~97 Hz against a learner that manages ~8.6 updates/s, so an operator's
    # correction is diluted across ~11 environment steps before any gradient
    # sees it -- and `steps_per_update` broadcasts weights only every ~5.8 s, by
    # which time the actor has moved on ~565 steps.  On real hardware physics
    # imposes this pacing for free; in sim it has to be asked for.
    #
    # `record_demo` and `test_intervention` resolve the same setting through the
    # same helper: a demo recorded at one rate and an intervention made at
    # another are different action distributions over identical physics, and the
    # learner samples 50/50 from both.
    rate_hz = resolve_rate_hz(workspace, FLAGS.actor_rate_hz)
    limiter = RateLimiter(rate_hz)
    if limiter.enabled:
        print_green(f"actor paced at {rate_hz:g} Hz")

    grasp_critic_enabled = _grasp_critic_enabled(workspace)

    pbar = tqdm.tqdm(range(workspace.max_steps), dynamic_ncols=True)
    step = 0  # bound up front: the `finally` flush reads it to name the dump
    try:
        for step in pbar:
            timer.tick("total")

            with timer.context("sample_actions"):
                if step < workspace.random_steps:
                    actions = env.action_space.sample()
                else:
                    with torch.no_grad(), parameter_lock:
                        obs_tensor = dict_apply(
                            obs, lambda x: torch.as_tensor(x, device=device)
                        )
                        actions = agent.sample_actions(
                            observations=obs_tensor,
                            argmax=False,
                        )
                    actions = actions.cpu().numpy()

            # Step environment
            with timer.context("step_env"):
                next_obs, reward, terminated, truncated, info = env.step(actions)
                reward = np.asarray(reward, dtype=np.float32)

                if "total_action" in info:
                    actions = info["total_action"]

                # Track intervention statistics
                if "intervene_action" in info:
                    actions = info["intervene_action"]
                    intervention_steps += 1
                    if not already_intervened:
                        intervention_count += 1
                    already_intervened = True
                else:
                    already_intervened = False
                actions = np.clip(actions, env.action_space.low, env.action_space.high)
                running_return += float(reward)

                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=np.asarray(1.0 - float(terminated), dtype=np.float32),
                    dones=bool(terminated or truncated),
                )
                # Only when the buffers were built with the column: a key the
                # buffer did not allocate is silently dropped, and a column the
                # transition lacks raises KeyError.
                if grasp_critic_enabled:
                    transition["grasp_penalty"] = np.asarray(
                        info.get("grasp_penalty", 0.0), dtype=np.float32
                    )
                
                # All data goes into replay buffer
                data_store.insert(transition)
                if dump_transitions:
                    transitions.append(copy.deepcopy(transition))

                # Intervention data additionally goes into intervention buffer
                if already_intervened:
                    intvn_data_store.insert(transition)
                    if dump_transitions:
                        demo_transitions.append(copy.deepcopy(transition))

                obs = next_obs
                if terminated or truncated:
                    # Add intervention statistics to episode info
                    if "episode" in info:
                        info["episode"]["intervention_count"] = intervention_count
                        info["episode"]["intervention_steps"] = intervention_steps
                    
                    stats = {"environment": info}
                    client.request("send-stats", stats)
                    pbar.set_description(f"Return: {running_return}")
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    client.update()
                    obs, _ = env.reset()
                    # A reset blocks for far longer than a step; without this
                    # the limiter reads it as a backlog and sprints the first
                    # steps of the new episode to catch up.
                    limiter.reset()

            if (
                dump_transitions
                and step > 0
                and step % workspace.buffer_period == 0
            ):
                # Hand the lists over and rebind immediately -- the writer owns
                # them from here, so this costs a reference assignment rather
                # than the 244 ms the inline pickle used to take.
                dump_writer.submit(
                    os.path.join(FLAGS.checkpoint_path, "buffer"), step, transitions
                )
                transitions = []

                if demo_transitions:
                    dump_writer.submit(
                        os.path.join(FLAGS.checkpoint_path, "demo_buffer"),
                        step,
                        demo_transitions,
                    )
                    demo_transitions = []

            timer.tock("total")

            if step % workspace.actor_update_period == 0:
                client.update()

            if step % workspace.log_period == 0:
                stats = {"timer": timer.get_average_times()}
                client.request("send-stats", stats)

            # Sleep after tocking, so the timers report work done rather than
            # time waited.  A no-op when unthrottled.
            limiter.sleep()
    finally:
        # Whatever accumulated since the last dump would otherwise be lost, and
        # the writer may still have a batch in flight.
        if dump_writer is not None:
            if transitions:
                dump_writer.submit(
                    os.path.join(FLAGS.checkpoint_path, "buffer"), step, transitions
                )
            if demo_transitions:
                dump_writer.submit(
                    os.path.join(FLAGS.checkpoint_path, "demo_buffer"),
                    step,
                    demo_transitions,
                )
            dump_writer.close()
        client.update()
        client.stop()
        env.close()


def _pretrain(agent, demo_buffer, device: str, start_step: int) -> int:
    """Behavior-clone the demo buffer, then checkpoint so the actor inherits it.

    Runs before the replay-buffer fill loop below, which is the point: that loop
    waits on *online* data, so without this the actor spends its first hundred
    steps driving a randomly initialised policy, and the demonstrations only
    ever reach the agent diluted 50/50 into online batches long afterwards.

    A demo-only iterator is mandatory here.  `next_training_batch` samples the
    replay buffer whenever the demo buffer is non-empty, and that buffer is
    still empty at this point -- `Dataset.sample` would index into a zero-length
    array and raise.
    """
    if len(demo_buffer) == 0:
        raise ValueError(
            "--pretrain_steps was given but the demo buffer is empty. Pass "
            "--demo_path=<file.pkl>."
        )

    iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": workspace.batch_size}, device=device
    )
    agent.train()
    pbar = tqdm.tqdm(range(FLAGS.pretrain_steps), desc="BC pretrain", dynamic_ncols=True)
    for step in pbar:
        info = agent.update_bc(next(iterator))
        # The critic is deliberately NOT fitted here, and that is a reversal of
        # what this loop used to do.  Measured on this task's 20 human demos:
        # 1500 critic updates on demo-only data drive Q from +0.58 to -5.2 with
        # `Q(demo) - Q(random)` going *negative* (-0.049), i.e. the critic ends
        # up preferring a random action to a demonstrated one.  The online actor
        # loss maximises that Q, so the first updates actively push the policy
        # away from the warm start -- which is exactly the "pretrain does
        # nothing, the arm goes back to flailing" symptom.
        #
        # The cause is extrapolation, not reward scale.  `_compute_next_actions`
        # evaluates the target at actions sampled from the *current* policy, and
        # on a demo-only buffer those actions are off-distribution: nothing in
        # the data constrains Q there, the unconstrained value is bootstrapped
        # back in, and the error compounds.  Adding `reward_bias=-1.0` makes it
        # worse rather than better (Q reaches -14.6 by the same step and is
        # still falling), because it scales up a value that is already
        # diverging.  Online data does not have this problem: the replay buffer
        # is drawn from the policy itself, so the target actions are in
        # distribution, which is why the same probe on a real run's step-5000
        # checkpoint reads +0.018 rather than negative.
        #
        # So the critic is left at its initialisation and learns online, where
        # its own action distribution is covered.  `update_bc` still warm-starts
        # the actor and the grasp head, which is measured to work: |mode| lands
        # at 0.069 against the demos' 0.081, and the grasp head stays
        # non-degenerate.
        if step % workspace.log_period == 0:
            pbar.set_postfix(bc=f"{info['bc_loss']:.4f}")

    step = start_step + FLAGS.pretrain_steps
    if FLAGS.checkpoint_path:
        os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
        # The filename has to match main()'s `checkpoint_[0-9]*.pt` glob, which
        # is how both the learner and the actor pick this up on the next launch.
        # Handing it over through the network broadcast instead would not work:
        # the initial publish happens after the fill loop, and a zmq subscriber
        # that has not connected yet misses whatever was published before it.
        path = os.path.join(FLAGS.checkpoint_path, f"checkpoint_{step}.pt")
        with torch.no_grad():
            torch.save({"step": step, "model_state_dict": agent.state_dict()}, path)
        print_green(f"Saved pretrained checkpoint to {path}")
    return step + 1


def learner(
    agent,
    replay_buffer: ReplayBufferDataStore,
    demo_buffer: ReplayBufferDataStore,
    device: str = "cuda",
    start_step: int = 0,
):
    agent.train()
    wandb_logger = make_wandb_logger(
        project="serl-plus-plus",
        description=FLAGS.exp_name,
        debug=FLAGS.debug,
    )

    # The grasp critic rides in the critic-only set as well as the full one, so
    # it gets the same cta_ratio as the continuous critic rather than half of it.
    critic_only_networks = frozenset({"critic"})
    all_networks = frozenset({"actor", "critic", "temperature"})
    if _grasp_critic_enabled(workspace):
        critic_only_networks = critic_only_networks | {"grasp_critic"}
        all_networks = all_networks | {"grasp_critic"}
        print_green("training a discrete grasp critic for the gripper channel")

    step = start_step
    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    if FLAGS.pretrain_steps > 0:
        start_step = _pretrain(agent, demo_buffer, device, start_step)
        # `stats_callback` closes over `step` to stamp the actor's episode stats,
        # and wandb rejects a step that goes backwards.
        step = start_step
        if FLAGS.pretrain_only:
            return

    pbar = tqdm.tqdm(
        total=workspace.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < workspace.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(state_dict_to_numpy(agent.state_dict()))
    print_green("sent initial network to actor")

    if start_step >= workspace.max_steps:
        # Silent otherwise: `range(start_step, max_steps)` is simply empty, so
        # the learner sits there having loaded everything and never takes a
        # gradient step while the actor waits for weights that never change.
        # Easy to hit now that --pretrain_steps counts towards the step number.
        raise ValueError(
            f"Nothing to train: resuming at step {start_step} but "
            f"training.max_steps is {workspace.max_steps}. Raise max_steps "
            "above the resumed step, or start from a fresh checkpoint path."
        )

    demo_batch_size = workspace.batch_size // 2
    online_batch_size = workspace.batch_size - demo_batch_size
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": workspace.batch_size,
        },
        device=device,
    )
    mixed_replay_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": online_batch_size},
        device=device,
    )
    demo_iterator = None

    def next_training_batch():
        nonlocal demo_iterator
        if len(demo_buffer) == 0:
            return next(replay_iterator)
        if demo_iterator is None:
            demo_iterator = demo_buffer.get_iterator(
                sample_args={"batch_size": demo_batch_size},
                device=device,
            )
        return concat_batches(
            next(mixed_replay_iterator),
            next(demo_iterator),
            axis=0,
        )

    timer = Timer()

    pbar = tqdm.tqdm(total=workspace.replay_buffer_capacity,
                     initial=len(replay_buffer), desc="replay buffer")

    for step in tqdm.tqdm(
        range(start_step, workspace.max_steps),
        dynamic_ncols=True,
        desc="learner",
    ):
        for _ in range(workspace.cta_ratio - 1):
            with timer.context("train_critics"):
                batch = next_training_batch()
                agent.update(batch, networks_to_update=critic_only_networks)

        with timer.context("train"):
            batch = next_training_batch()
            update_info = agent.update(batch,
                networks_to_update=all_networks
            )

        if step > 0 and step % (workspace.steps_per_update) == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            with torch.no_grad():
                state_dict = agent.state_dict()
                numpy_params = state_dict_to_numpy(state_dict)
            server.publish_network(numpy_params)
            del state_dict, numpy_params
            torch.cuda.empty_cache()

        if step % workspace.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            FLAGS.checkpoint_path
            and workspace.checkpoint_period
            and step > 0
            and step % workspace.checkpoint_period == 0
        ):
            os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
            checkpoint_file = os.path.join(FLAGS.checkpoint_path, f"checkpoint_{step}.pt")

            with torch.no_grad():
                torch.save({'step': step, 'model_state_dict': agent.state_dict()}, checkpoint_file)
            print_green(f"Saved checkpoint to {checkpoint_file}")
            torch.cuda.empty_cache()

        pbar.update(len(replay_buffer) - pbar.n)


def main(_):
    global workspace
    if not FLAGS.exp_name:
        choices = ", ".join(SERLWorkspace.available_tasks())
        raise ValueError(f"--exp_name is required. Available tasks: {choices}")
    if FLAGS.learner == FLAGS.actor:
        raise ValueError("Select exactly one of --learner or --actor")
    if FLAGS.eval_n_trajs < 0:
        raise ValueError("--eval_n_trajs must be non-negative")
    if FLAGS.eval_video_fps <= 0:
        raise ValueError("--eval_video_fps must be positive")
    if FLAGS.eval_video_start_hold_seconds < 0:
        raise ValueError("--eval_video_start_hold_seconds must be non-negative")
    if FLAGS.eval_video_end_hold_seconds < 0:
        raise ValueError("--eval_video_end_hold_seconds must be non-negative")
    if FLAGS.eval_n_trajs and not FLAGS.actor:
        raise ValueError("Evaluation requires --actor")
    if FLAGS.eval_n_trajs and not FLAGS.checkpoint_path:
        raise ValueError("Evaluation requires --checkpoint_path")

    workspace = SERLWorkspace(FLAGS.exp_name, FLAGS.config_override)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_green(f"Using device: {device}")
    
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    env = workspace.get_environment(fake_env=FLAGS.learner, seed=FLAGS.seed)
    env = RecordEpisodeStatistics(env)
    agent = workspace.get_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
    )
    agent = agent.to(device)

    start_step = 0
    loaded_checkpoint_step = None
    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        if FLAGS.eval_checkpoint_step > 0:
            requested_checkpoint = os.path.join(
                FLAGS.checkpoint_path,
                f"checkpoint_{FLAGS.eval_checkpoint_step}.pt",
            )
            checkpoint_files = (
                [requested_checkpoint] if os.path.isfile(requested_checkpoint) else []
            )
        else:
            checkpoint_files = glob.glob(
                os.path.join(FLAGS.checkpoint_path, "checkpoint_[0-9]*.pt")
            )
        if checkpoint_files:
            latest_checkpoint = max(
                checkpoint_files,
                key=lambda path: int(
                    os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[1]
                ),
            )
            ckpt = torch.load(latest_checkpoint, map_location=device)
            agent.load_state_dict(ckpt['model_state_dict'], strict=False)
            print_green(f"Loaded previous checkpoint at step {ckpt['step']} from {latest_checkpoint}.")
            loaded_checkpoint_step = int(ckpt["step"])
            start_step = loaded_checkpoint_step + 1
        else:
            if FLAGS.eval_n_trajs:
                requested = (
                    f"checkpoint_{FLAGS.eval_checkpoint_step}.pt"
                    if FLAGS.eval_checkpoint_step > 0
                    else "checkpoint_[step].pt"
                )
                raise FileNotFoundError(
                    f"No {requested} found in {FLAGS.checkpoint_path}"
                )
            print_green("Checkpoint directory exists but no checkpoint files found.")

    if FLAGS.eval_n_trajs:
        if loaded_checkpoint_step is None:
            raise FileNotFoundError(
                f"No checkpoint found at {FLAGS.checkpoint_path}"
            )
        try:
            evaluate(agent, env, str(device), loaded_checkpoint_step)
        finally:
            env.close()
        return

    if FLAGS.learner:
        # Both buffers must agree: `concat_batches` walks the demo batch's keys
        # and indexes the online one, so a column present in only one of them
        # raises KeyError at the learner's first step.
        include_grasp_penalty = _grasp_critic_enabled(workspace)
        replay_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=workspace.replay_buffer_capacity,
            device="cpu",
            include_grasp_penalty=include_grasp_penalty,
        )

        # The demo buffer only ever holds demonstrations plus interventions, so
        # sizing it like the online stream doubled the learner's image memory for
        # slots that never fill.  Read via getattr so a task config predating the
        # key still works -- it then falls back to the old shared capacity.
        demo_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=getattr(
                workspace, "demo_buffer_capacity", workspace.replay_buffer_capacity
            ),
            device="cpu",
            include_grasp_penalty=include_grasp_penalty,
        )
        print_green("replay buffer created")

        if FLAGS.demo_path:
            for path in FLAGS.demo_path:
                with open(path, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                            transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                        elif include_grasp_penalty:
                            # A demo recorded before the env emitted this key.
                            # `_insert_recursively` iterates the buffer's columns
                            # and would raise KeyError, so fail loudly with the
                            # actual remedy rather than on a bare key name.
                            raise ValueError(
                                f"{path} has no grasp_penalty in its infos, but "
                                f"{FLAGS.exp_name} trains a grasp critic. Re-record "
                                "the demos with the current code: "
                                f"./scripts/run_hil_serl.sh {FLAGS.exp_name} demos 20"
                            )
                        demo_buffer.insert(transition)
        else:
            print_green("No demo path provided. Starting with an empty demo buffer.")
        
        # Replayed buffer dumps were written by an actor, so they already carry
        # `grasp_penalty` at the top level when the feature was on -- but a
        # directory left over from a run that predates it would only surface as a
        # bare `KeyError: 'grasp_penalty'` from inside the buffer.
        def _load_dumps(directory, buffer, label):
            # Guard before the join, not after: os.path.join(None, ...) raises
            # TypeError, which is what a learner launched without a checkpoint
            # path used to die on.
            if FLAGS.checkpoint_path is None:
                return
            path = os.path.join(FLAGS.checkpoint_path, directory)
            if not os.path.exists(path):
                return
            files = sorted(glob.glob(os.path.join(path, "*.pkl")))
            loaded = 0
            for file in files:
                with open(file, "rb") as f:
                    for transition in pkl.load(f):
                        if include_grasp_penalty and "grasp_penalty" not in transition:
                            raise ValueError(
                                f"{file} predates the grasp critic. Delete "
                                f"{path} or start from a fresh --checkpoint_path."
                            )
                        buffer.insert(transition)
                        loaded += 1
            # A ring buffer wraps in silence, and for the demo buffer that means
            # overwriting the demonstrations loaded moments earlier with online
            # data -- the run then trains against a demo buffer that is not
            # demos, with nothing in the logs to say so.
            if loaded > buffer._capacity:
                raise ValueError(
                    f"{path} holds {loaded} transitions but the buffer's capacity "
                    f"is {buffer._capacity}; loading them would wrap and overwrite "
                    f"the earliest ones. Raise the matching *_buffer_capacity in "
                    f"config/task/{FLAGS.exp_name}.yaml, or delete {path}."
                )
            print_green(f"Loaded previous {label}. Size: {len(buffer)}")

        _load_dumps("buffer", replay_buffer, "buffer data")
        _load_dumps("demo_buffer", demo_buffer, "demo buffer data")

        print_green(f"replay_buffer size: {len(replay_buffer)}")
        print_green(f"demo buffer size: {len(demo_buffer)}")

        print_green("starting learner loop")
        learner(
            agent,
            replay_buffer=replay_buffer,
            demo_buffer=demo_buffer,
            device=device,
            start_step=start_step,
        )

    elif FLAGS.actor:
        # These are send queues, not replay buffers: `client.update()` drains them
        # every `actor_update_period` steps.  They were hardcoded at 50000, and
        # agentlace never drops transmitted data -- entries are evicted only when
        # the deque is full -- so the actor pinned up to 50000 transitions' worth
        # of images (~9 GiB) alongside the learner's own buffers on the same
        # machine.  A few thousand is already several minutes of backlog at these
        # rates; the deque discards oldest-first if the learner is unreachable for
        # longer than that, which is the right failure -- the alternative is
        # taking the desktop down with it.
        actor_queue_capacity = getattr(workspace, "actor_queue_capacity", 5000)
        data_store = QueuedDataStore(actor_queue_capacity)
        demo_data_store = QueuedDataStore(actor_queue_capacity)
        
        print_green("starting actor loop")
        actor(agent, data_store, demo_data_store, env, device=device)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)

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

    obs, _ = env.reset()
    timer = Timer()
    running_return = 0.0
    intervention_count = 0
    intervention_steps = 0
    already_intervened = False

    pbar = tqdm.tqdm(range(workspace.max_steps), dynamic_ncols=True)
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
                
                # All data goes into replay buffer
                data_store.insert(transition)
                transitions.append(copy.deepcopy(transition))
                
                # Intervention data additionally goes into intervention buffer
                if already_intervened:
                    intvn_data_store.insert(transition)
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

            if (
                FLAGS.checkpoint_path
                and step > 0
                and workspace.buffer_period > 0
                and step % workspace.buffer_period == 0
            ):
                buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
                os.makedirs(buffer_path, exist_ok=True)
                with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(transitions, f)
                    transitions = []

                if demo_transitions:
                    demo_buffer_path = os.path.join(
                        FLAGS.checkpoint_path, "demo_buffer"
                    )
                    os.makedirs(demo_buffer_path, exist_ok=True)
                    with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                        pkl.dump(demo_transitions, f)
                        demo_transitions = []

            timer.tock("total")

            if step % workspace.actor_update_period == 0:
                client.update()

            if step % workspace.log_period == 0:
                stats = {"timer": timer.get_average_times()}
                client.request("send-stats", stats)
    finally:
        client.update()
        client.stop()
        env.close()


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
                agent.update(batch, networks_to_update=frozenset({"critic"}))

        with timer.context("train"):
            batch = next_training_batch()
            update_info = agent.update(batch, 
                networks_to_update=frozenset({"actor", "critic", "temperature"})
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
        replay_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=workspace.replay_buffer_capacity,
            device="cpu"
        )
        
        demo_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=workspace.replay_buffer_capacity,
            device="cpu",
        )
        print_green("replay buffer created")

        if FLAGS.demo_path:
            for path in FLAGS.demo_path:
                with open(path, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                            transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                        demo_buffer.insert(transition)
        else:
            print_green("No demo path provided. Starting with an empty demo buffer.")
        
        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in sorted(
                glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in sorted(
                glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl"))
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}")

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
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        demo_data_store = QueuedDataStore(50000)
        
        print_green("starting actor loop")
        actor(agent, data_store, demo_data_store, env, device=device)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)

import warnings
warnings.filterwarnings("ignore")

import logging
import glob
logging.getLogger('asyncio').setLevel(logging.ERROR)

import time
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
from workspace import RLPDWorkspace


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
flags.DEFINE_boolean("debug", False, "Debug mode.")  


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
        choices = ", ".join(RLPDWorkspace.available_tasks())
        raise ValueError(f"--exp_name is required. Available tasks: {choices}")
    if FLAGS.learner == FLAGS.actor:
        raise ValueError("Select exactly one of --learner or --actor")

    workspace = RLPDWorkspace(FLAGS.exp_name, FLAGS.config_override)
    
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
    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        checkpoint_files = glob.glob(
            os.path.join(FLAGS.checkpoint_path, "checkpoint_[0-9]*.pt")
        )
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            ckpt = torch.load(latest_checkpoint, map_location=device)
            agent.load_state_dict(ckpt['model_state_dict'], strict=False)
            print_green(f"Loaded previous checkpoint at step {ckpt['step']} from {latest_checkpoint}.")
            start_step = int(ckpt["step"]) + 1
        else:
            print_green(f"Checkpoint directory exists but no checkpoint files found.")

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

import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags

from workspace import RLPDWorkspace

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_multi_string(
    "config_override",
    [],
    "Hydra task overrides, for example environment.config.random_reset=true.",
)
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_integer("seed", 42, "Environment seed.")

def main(_):
    workspace = RLPDWorkspace(FLAGS.exp_name, FLAGS.config_override)
    env = workspace.get_environment(fake_env=False, seed=FLAGS.seed)
    
    obs, info = env.reset(seed=FLAGS.seed)
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    
    while success_count < success_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, terminated, truncated, info = env.step(actions)
        episode_done = terminated or truncated
        returns += rew

        if "total_action" in info:
            actions = info["total_action"]
            
        if "intervene_action" in info:
            actions = info["intervene_action"]
            actions = np.clip(actions, env.action_space.low, 
                                       env.action_space.high)
        
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=np.asarray(1.0 - float(terminated), dtype=np.float32),
                dones=episode_done,
                infos=info,
            )
        )
        trajectory.append(transition)
        
        pbar.set_description(f"Return: {returns}")

        obs = next_obs
        if episode_done:
            if info["succeed"]:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)
            trajectory = []
            returns = 0
            obs, info = env.reset()
            
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")
    env.close()

if __name__ == "__main__":
    app.run(main)

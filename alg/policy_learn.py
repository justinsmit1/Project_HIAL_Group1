from time import sleep
from envs.task_envs import PnPNewRobotEnv
from utils.env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper
import csv
from pathlib import Path
import numpy as np
import sys
import gymnasium as gym
from utils.env_wrappers import (
    ActionNormalizer,
    ResetWrapper,
    TimeLimitWrapper,
    reconstruct_state,
)
from utils.demos import prepare_demo_pool


sys.path.append("AWAC")
from alg.awac import AWAC, ReplayBuffer

# @software{Sikchi_pytorch-AWAC,
# author = {Sikchi, Harshit and Wilcox, Albert},
# doi = {10.5281/zenodo.5121023},
# title = {{pytorch-AWAC}},
# url = {https://github.com/hari-sikchi/AWAC}
# }

# 1. Load feature weights from CSV
# 2. Create the environment
# 3. Create the AWAC agent
# 4. Load expert demos into replay buffer
# 5. Training loop (500k steps):
#    a. Roll out current policy for one episode
#    b. Compute rewards using comp_reward
#    c. Store transitions in replay buffer
#    d. Update the agent
#    e. Every 1k steps: evaluate and save
# 6. Plot learning curve


# load weights
repo_root = Path(__file__).resolve().parents[1]
weights_path = repo_root / "saved" / "feature_weights.csv"
weights = []
with open(weights_path) as fw:
    reader = csv.DictReader(fw)
    for row in reader:
        weights.append(float(row["weight"]))
weights = np.array(weights)
# print(weights)


# reward function
def comp_reward(t, dists, weights, success, num_steps):
    per_step = weights[0] * (-dists[t]) + weights[3]
    terminal = weights[4] if success else 0
    return per_step + (terminal if t == (num_steps - 1) else 0)


# create awac agent
# but AWAC in the init takes env_fn not an env itself because it makes a copy of the environment
# chatGPT made wrapper
class FlattenObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return reconstruct_state(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return reconstruct_state(obs), reward, terminated, truncated, info


# create the environment
def make_env():
    env = PnPNewRobotEnv(render=False)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    env = FlattenObsWrapper(env)
    return env


def main() -> None:
    # create the AWAC agent
    agent = AWAC(env_fn=make_env)

    # load expert demos into replay buffer
    demo_dir = repo_root / "demo_data" / "PickAndPlace"
    demos = prepare_demo_pool(demo_dir, verbose=True)

    for demo in demos:
        states = demo["state_trajectory"]
        actions = demo["action_trajectory"]
        next_states = demo["next_state_trajectory"]
        rewards = demo["reward_trajectory"]
        dones = demo["done_trajectory"]
        T = len(actions)

        for t in range(T):
            agent.replay_buffer.store(
                states[t], actions[t], rewards[t], next_states[t], dones[t]
            )

    # check demo state
    # print(f"Replay buffer size after demos: {agent.replay_buffer.size}")
    # print("demo state shape:", states[0].shape)
    # print("demo state:", states[0])
    # check live env state

    # untested from here on
    # training loop
    total_steps = 0
    max_steps = 500000
    while total_steps < max_steps:
        obs, info = agent.env.reset()
        done = False
        currEpisode_states = [obs]
        currEpisode_actions = []
        currEpisode_dones = []

        while not done:
            action = agent.get_action(obs)

            next_obs, _, terminated, truncated, info = agent.env.step(action)
            done = terminated or truncated

            currEpisode_states.append(next_obs)
            currEpisode_actions.append(action)
            currEpisode_dones.append(done)
            obs = next_obs
            total_steps += 1

        T = len(currEpisode_actions)


if __name__ == "__main__":
    main()

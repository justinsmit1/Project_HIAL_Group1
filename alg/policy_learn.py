import os
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
import torch
from alg.awac import AWAC, ReplayBuffer
import matplotlib.pyplot as plt

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
def comp_reward(t, dists, arm_dists, weights, episode_success, num_steps):
    # reward for banana getting close to plate
    banana_to_plate = weights[0] * (-dists[t])
    # reward for arm getting close to banana (helps agent learn to pick up)
    arm_to_banana = -arm_dists[t]
    # step penalty
    step_penalty = weights[3]
    # terminal bonus
    terminal = weights[4] if (t == num_steps - 1 and episode_success) else 0

    per_step = banana_to_plate + arm_to_banana + step_penalty
    return per_step + terminal


# create awac agent


# create the environment
def make_env():
    env = PnPNewRobotEnv(render=False)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    env.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
    )

    return env


#   [0:3]   ee_pos
#   [3:6]   ee_vel
#   [6]     finger_width
#   [7:10]  banana_pos
#   [10:14] banana_quat
#   [14:17] banana_vel
#   [17:20] banana_ang_vel
#   [20:23] desired_goal (plate center)
def compute_distances(flat_states, T):
    goal_start = flat_states[0].shape[0] - 3

    dists = np.array(
        [
            np.linalg.norm(
                flat_states[t][7:10] - flat_states[t][goal_start : goal_start + 3]
            )
            for t in range(T)
        ]
    )
    arm_dists = np.array(
        [np.linalg.norm(flat_states[t][0:3] - flat_states[t][7:10]) for t in range(T)]
    )
    return dists, arm_dists


def main() -> None:
    os.makedirs("saved/policy_learning_curve_steps", exist_ok=True)
    # create the AWAC agent
    agent = AWAC(env_fn=make_env)

    # load expert demos into replay buffer and store
    demo_dir = repo_root / "demo_data" / "PickAndPlace"
    demos = prepare_demo_pool(demo_dir, verbose=True)

    for demo in demos:
        states = demo["state_trajectory"]
        actions = demo["action_trajectory"]
        next_states = demo["next_state_trajectory"]
        rewards = demo["reward_trajectory"]
        dones = demo["done_trajectory"]
        T = len(actions)

        dists, arm_dists = compute_distances(states, T)
        episode_success = bool(np.squeeze(dones[-1]))

        for t in range(T):
            reward = comp_reward(t, dists, arm_dists, weights, episode_success, T)
            agent.replay_buffer.store(
                states[t],
                actions[t],
                reward,
                next_states[t],
                float(np.squeeze(dones[t])),
            )

    # check demo state
    print(f"Replay buffer size after demos: {agent.replay_buffer.size}")
    # print("demo state shape:", states[0].shape)
    # print("demo state:", states[0])
    # check live env state

    # untested from here on
    # training loop
    # 5. Training loop (500k steps):
    #    a. Roll out current policy for one episode
    #    b. Compute rewards using comp_reward
    #    c. Store transitions in replay buffer
    #    d. Update the agent
    #    e. Every 1k steps: evaluate and save
    # 6. Plot learning curve

    # state 7:10 = banana position
    # state 19:22 = goal position, subtract these

    # print("agent.env", agent.env)
    # print("demo state, ", states)
    # print("demo state[7], ", states[7])
    # print(states[0][19:])
    total_steps = 0
    max_steps = 15000
    last_save = 0
    steps = []
    success_rates = []
    eval_env = make_env()

    while total_steps < max_steps:
        obs_dict, info = agent.env.reset()
        obs = reconstruct_state(obs_dict) if isinstance(obs_dict, dict) else obs_dict
        done = False
        currEpisode_states = [obs]
        currEpisode_actions = []
        currEpisode_info = []

        while not done:
            action = agent.get_action(obs)

            next_obs_dict, _, terminated, truncated, info = agent.env.step(action)
            next_obs = (
                reconstruct_state(next_obs_dict)
                if isinstance(next_obs_dict, dict)
                else next_obs_dict
            )
            done = terminated or truncated

            currEpisode_states.append(next_obs)
            currEpisode_actions.append(action)
            currEpisode_info.append(info)
            obs = next_obs
            total_steps += 1
            # print(f"total_steps: {total_steps}")

        T = len(currEpisode_actions)
        # print("currEpisode_states", currEpisode_states)

        # test
        episode_success = bool(currEpisode_info[-1].get("is_success", False))

        print(
            f"Step {total_steps} | Train ep success: {episode_success} | final info: {currEpisode_info[-1]}"
        )

        dists, arm_dists = compute_distances(currEpisode_states, T)

        for t in range(T):
            # currEpisode_dones is bool of wether the episode finished at timestep t or not
            reward = comp_reward(t, dists, arm_dists, weights, episode_success, T)

            # store transitions in replay buffer
            agent.replay_buffer.store(
                currEpisode_states[t],
                currEpisode_actions[t],
                reward,
                currEpisode_states[t + 1],
                float(t == T - 1),
            )

        # update agent
        if agent.replay_buffer.size > agent.batch_size:
            batch = agent.replay_buffer.sample_batch(agent.batch_size)
            agent.update(data=batch, update_timestep=total_steps)

        # print(f"Loss Q: {agent.compute_loss_q(data=batch)}, Loss Pi: {agent.compute_loss_pi(data=batch)}")

        # every 1k eval and save
        # had to do this way because it was skipping over 1000 with the 150 episode so total_steps % 1000 was never hitting
        if total_steps - last_save >= 1000:
            # agent.ac.state_dict() returns all the NN weights & biases as a dictionary
            torch.save(
                agent.ac.state_dict(),
                f"saved/policy_learning_curve_steps/policy_{total_steps}.pt",
            )
            last_save = total_steps
            print(f"Saved policy at step {total_steps}")

            successes = 0
            for _ in range(10):
                obs_dict, _ = eval_env.reset()
                obs_eval = reconstruct_state(obs_dict)
                terminated = truncated = False

                while not (terminated or truncated):
                    action = agent.get_action(obs_eval, deterministic=True)
                    next_obs_dict, _, terminated, truncated, eval_info = eval_env.step(
                        action
                    )
                    obs_eval = reconstruct_state(next_obs_dict)

                # Call is_success manually to see what it should return
                achieved_goal = eval_env.unwrapped.task.get_achieved_goal()
                # Object position
                desired_goal = eval_env.unwrapped.task.get_goal()  # Target position
                dist = np.linalg.norm(achieved_goal - desired_goal)
                print(
                    f"Step {total_steps} | Eval ep dist: {dist:.4f} | threshold: {eval_env.unwrapped.task.distance_threshold}"
                )
                # print(f"Manual is_success call: {manual_success}")
                # print(f"Distance: {np.linalg.norm(achieved_goal - desired_goal)}")
                # print(f"Threshold: {eval_env.unwrapped.task.distance_threshold}")

                if eval_info.get("is_success", False):
                    successes += 1

            success_rate = successes / 10
            steps.append(total_steps)
            success_rates.append(success_rate)

    if steps and success_rates:
        fix, ax = plt.subplots()
        ax.plot(steps, success_rates)
        ax.set_xlabel("Environment Steps", fontsize=12)
        ax.set_ylabel("Average Success Rate", fontsize=12)
        plt.savefig("saved/learning_curve.png")

    # need to fix success rate rollouts because its not working


if __name__ == "__main__":
    main()

import os
import csv
import json
import imageio
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Internal Imports
from envs.task_envs import PnPNewRobotEnv
from utils.env_wrappers import (
    ActionNormalizer,
    ResetWrapper,
    TimeLimitWrapper,
    reconstruct_state,
)
from utils.demos import prepare_demo_pool
from alg.awac import AWAC

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
MAX_STEPS = 100000
PRETRAIN_STEPS = 10000
EVAL_EVERY = 2000
BATCH_SIZE = 256
TABLE_Z = 0.4

repo_root = Path(__file__).resolve().parents[1]
weights_path = repo_root / "saved" / "feature_weights_volume_removal.csv"
output_dir = repo_root / "saved" / "policy_learning"
clips_dir = output_dir / "clips"
os.makedirs(clips_dir, exist_ok=True)


# ==========================================
# 2. IMPROVED REWARD & METRICS
# ==========================================
def compute_trajectory_metrics(flat_states: List[np.ndarray]):
    """
    Indices: 0:3=Gripper, 3=Finger Width, 7:10=Object XYZ, Last 3=Goal XYZ.
    """
    goal_idx = flat_states[0].shape[0] - 3

    obj_to_goal = np.array([np.linalg.norm(s[7:10] - s[goal_idx:]) for s in flat_states])
    grip_to_obj = np.array([np.linalg.norm(s[0:3] - s[7:10]) for s in flat_states])
    obj_heights = np.array([s[9] for s in flat_states])
    grip_widths = np.array([s[3] for s in flat_states])
    obj_positions = np.array([s[7:10] for s in flat_states])

    return obj_to_goal, grip_to_obj, obj_heights, grip_widths, obj_positions


import numpy as np

def comp_reward(t, metrics, weights, success, T_ep):
    """
    Improved reward for pick-and-place tasks.

    Args:
        t: time step
        metrics: obj_to_goal, grip_to_obj, obj_heights, grip_widths, obj_positions
        weights: array of feature weights [unused here except step penalty]
        success: bool, whether trajectory was successful
        T_ep: int, trajectory length
    """
    obj_to_goal, grip_to_obj, obj_heights, grip_widths, obj_positions = metrics

    reward = 0.0

    # --- 1. Reach reward (continuous) ---
    # Encourages gripper to get closer to the object
    reach_reward = np.exp(-10.0 * grip_to_obj[t])  # smooth exponential decay
    reward += 2.0 * reach_reward

    # --- 2. Grasping reward (continuous) ---
    # Encourage gripper to be near object AND partially closed
    grasp_factor = np.clip(1.0 - grip_widths[t] / 0.05, 0, 1)  # normalized closure
    near_factor = np.clip(1.0 - grip_to_obj[t] / 0.05, 0, 1)     # normalized proximity
    grasp_reward = near_factor * grasp_factor
    reward += 5.0 * grasp_reward

    # --- 3. Lift reward ---
    lift_height = max(0.0, obj_heights[t] - TABLE_Z)
    lift_reward = lift_height * grasp_reward  # only reward if grasped
    reward += 5.0 * lift_reward

    # --- 4. Move object toward goal ---
    if t > 0:
        prev_to_goal = obj_to_goal[t - 1]
        curr_to_goal = obj_to_goal[t]
        goal_progress = np.clip(prev_to_goal - curr_to_goal, 0, 0.1)  # reward forward progress
        reward += 5.0 * goal_progress

    # --- 5. Step penalty (small negative reward to encourage efficiency) ---
    reward += weights[3]  # usually negative

    # --- 6. Terminal success bonus ---
    if t == T_ep - 1 and success:
        reward += 20.0  # moderate bonus

    return reward

# def comp_reward(t, metrics, weights, success, T_ep):
#     obj_to_goal, grip_to_obj, obj_heights, grip_widths, obj_positions = metrics
#     reward = 0.0
#
#     # 1. Reach Reward (Sharper exponential to encourage precision)
#     dist_temp = 10 #15,
#     reward += 1.0 * np.exp(-dist_temp * grip_to_obj[t]) #1.0
#
#     # 2. Grasping Logic
#     # We want the gripper to be close AND the width to be small (closed)
#     is_near = grip_to_obj[t] < 0.04 #0.035
#     is_closed = grip_widths[t] < 0.03 #0.03
#
#     if is_near:
#         #reward += 2.0 * (1.0 - grip_widths[t])
#
#         if is_closed:
#             reward += 10.0  # Reward for actually closing while near
#         else:
#             reward -= 3  # Penalty for hovering with open claws, instead of -1
#
#     # 3. Lifting Reward (Only if we are actually grasping)
#     lift_height = max(0.0, obj_heights[t] - TABLE_Z)
#     if is_closed and is_near:
#         reward += 10.0 * lift_height
#
#     # 4. Movement & Goal
#     if is_near and is_closed:
#         # Move object toward goal
#         reward += weights[0] * (1.0 - np.tanh(4.0 * obj_to_goal[t])) #1.0
#
#         # Reward velocity toward goal
#         if t > 0:
#             vel = obj_to_goal[t - 1] - obj_to_goal[t]
#             reward += 20.0 * max(0, vel)
#
#     # 5. Penalties and Success
#     reward += weights[3]  # Small step penalty (usually negative)
#
#     if t == T_ep - 1 and success:
#         reward += 50.0  # Significant terminal bonus
#
#     return reward


# ==========================================
# 3. ENVIRONMENT & HELPERS
# ==========================================
def make_env(render=False):
    env = PnPNewRobotEnv(render=render)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
    return env


def capture_frame(env: Any):
    try:
        base = getattr(env, "unwrapped", env)
        pc = base.sim.physics_client
        view_matrix = pc.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 0.0], distance=1.2, yaw=45, pitch=-35, roll=0, upAxisIndex=2
        )
        proj_matrix = pc.computeProjectionMatrixFOV(fov=60.0, aspect=1.33, nearVal=0.01, farVal=10.0)
        _, _, px, _, _ = pc.getCameraImage(640, 480, view_matrix, proj_matrix, renderer=pc.ER_BULLET_HARDWARE_OPENGL)
        return np.reshape(px, (480, 640, 4))[:, :, :3].astype(np.uint8)
    except:
        return np.zeros((480, 640, 3), dtype=np.uint8)


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    w_list = []
    with open(weights_path) as f:
        reader = csv.DictReader(f)
        for row in reader: w_list.append(float(row["weight"]))
    weights = np.array(w_list)

    agent = AWAC(env_fn=lambda: make_env(render=False), batch_size=BATCH_SIZE)
    eval_env = make_env(render=False)

    # --- PART A: OFFLINE DEMO LOADING ---
    print("[1/3] Processing Expert Demos...")
    demo_env = make_env()
    demos = prepare_demo_pool(repo_root / "demo_data" / "PickAndPlace", verbose=False)

    for demo in demos:
        action_seq = demo["action_trajectory"]
        obs_dict, _ = demo_env.reset(seed=0)
        states, actions, next_states = [], [], []

        for act in action_seq:
            s_flat = reconstruct_state(obs_dict)
            next_obs_dict, _, term, trunc, info = demo_env.step(act)
            ns_flat = reconstruct_state(next_obs_dict)
            states.append(s_flat);
            actions.append(act);
            next_states.append(ns_flat)
            obs_dict = next_obs_dict
            if term or trunc: break

        if len(actions) == 0: continue
        metrics = compute_trajectory_metrics(states + [next_states[-1]])
        success = info.get("is_success", False)
        for t in range(len(actions)):
            r = comp_reward(t, metrics, weights, success, len(actions))
            agent.replay_buffer.store(states[t], actions[t], r, next_states[t], float(t == len(actions) - 1))

    # --- PART B: OFFLINE PRE-TRAINING ---
    print(f"[2/3] Offline Pre-training ({PRETRAIN_STEPS} steps)...")
    for i in range(PRETRAIN_STEPS):
        batch = agent.replay_buffer.sample_batch(BATCH_SIZE)
        agent.update(data=batch, update_timestep=i)

    # --- PART C: ONLINE TRAINING ---
    print(f"[3/3] Starting Online Training...")
    total_steps, steps_log, reward_log = 0, [], []

    while total_steps < MAX_STEPS:
        obs_dict, _ = agent.env.reset()
        obs = reconstruct_state(obs_dict)
        ep_history = []
        done = False

        while not done:
            action = agent.get_action(obs, deterministic=False)
            next_obs_dict, _, term, trunc, info = agent.env.step(action)
            next_obs = reconstruct_state(next_obs_dict)
            ep_history.append((obs, action, next_obs, term or trunc, info))
            obs = next_obs
            done = term or trunc
            total_steps += 1

        T_ep = len(ep_history)
        states_seq = [step[0] for step in ep_history] + [ep_history[-1][2]]
        metrics = compute_trajectory_metrics(states_seq)
        success = ep_history[-1][4].get("is_success", False)

        ep_reward_sum = 0
        for t in range(T_ep):
            s, a, ns, d, _ = ep_history[t]
            r = comp_reward(t, metrics, weights, success, T_ep)
            ep_reward_sum += r
            agent.replay_buffer.store(s, a, r, ns, float(d))

        # Gradient Updates
        if agent.replay_buffer.size > BATCH_SIZE:
            for _ in range(T_ep):
                batch = agent.replay_buffer.sample_batch(BATCH_SIZE)
                agent.update(data=batch, update_timestep=total_steps)

        # Periodic Evaluation
        if total_steps % EVAL_EVERY < T_ep:
            total_eval_reward = 0
            successes = 0
            frames = []

            for trial in range(10):
                o_d, _ = eval_env.reset()
                o = reconstruct_state(o_d)
                d_eval = False
                trial_reward = 0
                eval_states = [o]

                while not d_eval:
                    if trial == 0: frames.append(capture_frame(eval_env))
                    a = agent.get_action(o, deterministic=True)
                    o_d, _, te, tr, inf = eval_env.step(a)
                    o = reconstruct_state(o_d)
                    eval_states.append(o)
                    d_eval = te or tr

                # Compute reward for the evaluation trajectory
                eval_metrics = compute_trajectory_metrics(eval_states)
                trial_success = inf.get("is_success", False)
                if trial_success: successes += 1
                for t_eval in range(len(eval_states) - 1):
                    trial_reward += comp_reward(t_eval, eval_metrics, weights, trial_success, len(eval_states) - 1)
                total_eval_reward += trial_reward

            avg_reward = total_eval_reward / 10
            sr = successes / 10
            steps_log.append(total_steps)
            reward_log.append(avg_reward)

            print(f"Step: {total_steps:6d} | Avg Reward: {avg_reward:8.2f} | Success Rate: {sr:.2f}")

            if frames: imageio.mimsave(clips_dir / f"step_{total_steps}.mp4", frames, fps=30)
            torch.save(agent.ac.state_dict(), output_dir / f"policy_latest.pt")

    plt.figure(figsize=(10, 5))
    plt.plot(steps_log, reward_log)
    plt.title("Learning Curve")
    plt.xlabel("Steps");
    plt.ylabel("Average Episode Reward")
    plt.savefig(output_dir / "reward_curve.png")


if __name__ == "__main__":
    main()
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

# Internal Imports (Assuming standard project structure)
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
MAX_STEPS = 20000  # Total online environment steps
PRETRAIN_STEPS = 10000  # AWAC offline priming steps
EVAL_EVERY = 2000  # Eval interval & video capture
BATCH_SIZE = 256
TABLE_Z = 0.4  # Z-height of the table for lifting check

repo_root = Path(__file__).resolve().parents[1]
weights_path = repo_root / "saved" / "feature_weights_volume_removal.csv"
output_dir = repo_root / "saved" / "policy_learning"
clips_dir = output_dir / "clips"
os.makedirs(clips_dir, exist_ok=True)


# ==========================================
# 2. ENHANCED DISTANCE & REWARD LOGIC
# ==========================================
def compute_trajectory_metrics(flat_states: List[np.ndarray]):
    """
    Extracts metrics for reward calculation.
    Indices: 0:3=Gripper, 3=Finger Width, 7:10=Object XYZ, Last 3=Goal XYZ.
    """
    T = len(flat_states) - 1
    goal_idx = flat_states[0].shape[0] - 3

    # Pre-calculate arrays for the whole trajectory
    obj_to_goal = np.array([np.linalg.norm(s[7:10] - s[goal_idx:]) for s in flat_states])
    grip_to_obj = np.array([np.linalg.norm(s[0:3] - s[7:10]) for s in flat_states])
    obj_heights = np.array([s[9] for s in flat_states])
    grip_widths = np.array([s[3] for s in flat_states])

    return obj_to_goal, grip_to_obj, obj_heights, grip_widths


def comp_reward(t, metrics, weights, success, T_ep):
    """
    Multi-stage Dense Reward:
    1. Reaching (Exponential)
    2. Grasping (Finger closure near object)
    3. Lifting (Object Z > Table Z)
    4. Placing (Goal seeking - active only if lifted/grasped)
    """
    obj_to_goal, grip_to_obj, obj_heights, grip_widths = metrics
    reward = 0.0

    # Stage 1: Reach the banana
    reward += 2.0 * np.exp(-10.0 * grip_to_obj[t])

    # Stage 2: Grasp logic (width < 0.02 is closed)
    is_grasping = (grip_to_obj[t] < 0.04) and (grip_widths[t] < 0.02)
    if is_grasping:
        reward += 2.0

    # Stage 3: Lift check
    lifted = obj_heights[t] > (TABLE_Z + 0.05)
    if lifted:
        reward += 5.0

    # Stage 4: Place (Distance to goal) - Gated by possession
    if lifted or is_grasping:
        # Use tanh for a smooth reward curve as we approach goal
        reward += weights[0] * (1.0 - np.tanh(5.0 * obj_to_goal[t]))

    # Penalty and Terminal Bonus
    reward += weights[3]  # Step penalty
    if t == T_ep - 1 and success:
        reward += weights[4]  # Success bonus

    return reward


# ==========================================
# 3. ENVIRONMENT & VIDEO HELPERS
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
    # Load weights from CSV
    w_list = []
    with open(weights_path) as f:
        reader = csv.DictReader(f)
        for row in reader: w_list.append(float(row["weight"]))
    weights = np.array(w_list)

    agent = AWAC(env_fn=lambda: make_env(render=False), batch_size=BATCH_SIZE)
    eval_env = make_env(render=False)

    # --- PART A: OFFLINE DEMO LOADING ---
    print("[1/3] Loading Expert Demos...")
    demo_env = make_env()
    demos = prepare_demo_pool(repo_root / "demo_data" / "PickAndPlace", verbose=True)

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

        T_ep = len(actions)
        if T_ep == 0: continue

        metrics = compute_trajectory_metrics(states + [next_states[-1]])
        success = info.get("is_success", False)
        for t in range(T_ep):
            r = comp_reward(t, metrics, weights, success, T_ep)
            agent.replay_buffer.store(states[t], actions[t], r, next_states[t], float(t == T_ep - 1))

    # --- PART B: OFFLINE PRE-TRAINING ---
    print(f"[2/3] Offline Pre-training ({PRETRAIN_STEPS} steps)...")
    for i in range(PRETRAIN_STEPS):
        batch = agent.replay_buffer.sample_batch(BATCH_SIZE)
        agent.update(data=batch, update_timestep=i)

    # --- PART C: ONLINE TRAINING ---
    print(f"[3/3] Online Training...")
    total_steps, steps_log, success_log = 0, [], []

    while total_steps < MAX_STEPS:
        obs_dict, _ = agent.env.reset()
        obs = reconstruct_state(obs_dict)

        ep_history = []
        done = False
        while not done:
            action = agent.get_action(obs, deterministic=False)  # Explore
            next_obs_dict, _, term, trunc, info = agent.env.step(action)
            next_obs = reconstruct_state(next_obs_dict)

            ep_history.append((obs, action, next_obs, term or trunc, info))
            obs = next_obs
            done = term or trunc
            total_steps += 1

        # Calculate rewards for episode
        T_ep = len(ep_history)
        states_seq = [step[0] for step in ep_history] + [ep_history[-1][2]]
        metrics = compute_trajectory_metrics(states_seq)
        success = ep_history[-1][4].get("is_success", False)

        for t in range(T_ep):
            s, a, ns, d, _ = ep_history[t]
            r = comp_reward(t, metrics, weights, success, T_ep)
            agent.replay_buffer.store(s, a, r, ns, float(d))

        # Gradient Updates
        if agent.replay_buffer.size > BATCH_SIZE:
            for _ in range(T_ep):
                batch = agent.replay_buffer.sample_batch(BATCH_SIZE)
                agent.update(data=batch, update_timestep=total_steps)

        # Periodic Evaluation & Video
        if total_steps % EVAL_EVERY < T_ep:
            successes = 0
            frames = []
            for trial in range(10):
                o_d, _ = eval_env.reset()
                o = reconstruct_state(o_d)
                d_eval = False
                while not d_eval:
                    if trial == 0: frames.append(capture_frame(eval_env))
                    a = agent.get_action(o, deterministic=True)
                    o_d, _, te, tr, inf = eval_env.step(a)
                    o = reconstruct_state(o_d);
                    d_eval = te or tr
                if inf.get("is_success", False): successes += 1

            sr = successes / 10
            steps_log.append(total_steps);
            success_log.append(sr)
            print(f"Step {total_steps} | Success: {sr:.2f}")
            if frames: imageio.mimsave(clips_dir / f"step_{total_steps}.mp4", frames, fps=30)
            torch.save(agent.ac.state_dict(), output_dir / f"policy_{total_steps}.pt")

    # Plot final curve
    plt.plot(steps_log, success_log)
    plt.xlabel("Steps");
    plt.ylabel("Success Rate");
    plt.savefig(output_dir / "curve.png")


if __name__ == "__main__":
    main()
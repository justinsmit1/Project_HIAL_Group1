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
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
MAX_STEPS = 20000  # Total online environment steps
PRETRAIN_STEPS = 10000  # Offline iterations on demo data
EVAL_EVERY = 2000  # How often to run evaluation and save video
BATCH_SIZE = 256  # Transitions per gradient update
SAVE_VIDEO = True  # Enable MP4 generation during eval

repo_root = Path(__file__).resolve().parents[1]
weights_path = repo_root / "saved" / "feature_weights_volume_removal.csv"
output_dir = repo_root / "saved" / "policy_learning"
clips_dir = output_dir / "clips"

os.makedirs(clips_dir, exist_ok=True)


# ==========================================
# 2. REWARD & DISTANCE UTILITIES
# ==========================================
def load_weights():
    """Loads weights from CSV or defaults to sensible PnP values."""
    if weights_path.exists():
        weights = []
        with open(weights_path) as fw:
            reader = csv.DictReader(fw)
            for row in reader:
                weights.append(float(row["weight"]))
        return np.array(weights)
    else:
        # Fallback: [Obj2Goal, FinalDist, MinDist, StepPenalty, SuccessBonus]
        return np.array([1.5, 2.0, 0.5, -0.01, 20.0])


weights = load_weights()


def compute_distances(flat_states: List[np.ndarray], T: int):
    """
    Computes distances based on specific state vector indices.
    Indices 0:3 = Gripper, 7:10 = Object, Last 3 = Goal.
    """
    goal_start = flat_states[0].shape[0] - 3

    # Object to Goal
    dists = np.array([
        np.linalg.norm(flat_states[t][7:10] - flat_states[t][goal_start:goal_start + 3])
        for t in range(T)
    ])
    # Gripper to Object
    arm_dists = np.array([
        np.linalg.norm(flat_states[t][0:3] - flat_states[t][7:10])
        for t in range(T)
    ])
    return dists, arm_dists


def comp_reward(t, dists, arm_dists, weights, episode_success, num_steps):
    """Calculates the dense reward for a single timestep."""
    # Scaling factor to make small movements (cm) visible to the optimizer
    scale = 10.0

    obj_to_goal_rew = weights[0] * (-dists[t] * scale)
    arm_to_obj_rew = -arm_dists[t] * scale
    step_penalty = weights[3]

    terminal_bonus = 0
    if t == num_steps - 1 and episode_success:
        terminal_bonus = weights[4]

    return obj_to_goal_rew + arm_to_obj_rew + step_penalty + terminal_bonus


# ==========================================
# 3. ENVIRONMENT & VISUALIZATION
# ==========================================
def make_env(render=False):
    env = PnPNewRobotEnv(render=render)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    # Ensure obs space matches the flattened 22-dim vector used by AWAC
    env.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
    )
    return env


def capture_frame(env: Any, width: int = 480, height: int = 360) -> np.ndarray:
    """Renders the robot scene to an RGB array using PyBullet."""
    try:
        base = getattr(env, "unwrapped", env)
        pc = base.sim.physics_client
        view_matrix = pc.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 0.0],
            distance=1.2, yaw=45, pitch=-35, roll=0, upAxisIndex=2
        )
        proj_matrix = pc.computeProjectionMatrixFOV(
            fov=60.0, aspect=float(width) / float(height), nearVal=0.01, farVal=10.0
        )
        _, _, px, _, _ = pc.getCameraImage(
            width=width, height=height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pc.ER_BULLET_HARDWARE_OPENGL
        )
        rgba = np.reshape(px, (height, width, 4))
        return rgba[:, :, :3].astype(np.uint8)
    except Exception:
        return np.zeros((height, width, 3), dtype=np.uint8)


# ==========================================
# 4. MAIN TRAINING PIPELINE
# ==========================================
def main():
    # Initialize Agent
    agent = AWAC(env_fn=lambda: make_env(render=False), batch_size=BATCH_SIZE)
    eval_env = make_env(render=False)

    # --- STEP 1: LOAD DEMONSTRATIONS ---
    demo_dir = repo_root / "demo_data" / "PickAndPlace"
    demos = prepare_demo_pool(demo_dir, verbose=True)

    print(f"\n[1/4] Loading {len(demos)} demos into Replay Buffer...")
    demo_env = make_env()

    for demo in demos:
        action_seq = demo["action_trajectory"]
        obs_dict, _ = demo_env.reset(seed=0)

        temp_states, temp_actions, temp_next_states = [], [], []

        # Rollout demo actions to get correct next_states and rewards
        for action in action_seq:
            curr_state_flat = reconstruct_state(obs_dict)
            next_obs_dict, _, terminated, truncated, info = demo_env.step(action)
            next_state_flat = reconstruct_state(next_obs_dict)

            temp_states.append(curr_state_flat)
            temp_actions.append(action)
            temp_next_states.append(next_state_flat)

            obs_dict = next_obs_dict
            if terminated or truncated: break

        T = len(temp_actions)
        if T == 0: continue

        # Calculate rewards based on the trajectory just generated
        states_for_dist = temp_states + [temp_next_states[-1]]
        dists, arm_dists = compute_distances(states_for_dist, T)
        success = info.get("is_success", False)

        for t in range(T):
            reward = comp_reward(t, dists, arm_dists, weights, success, T)
            agent.replay_buffer.store(
                temp_states[t], temp_actions[t], reward, temp_next_states[t], float(t == T - 1)
            )

    # --- STEP 2: OFFLINE PRE-TRAINING ---
    print(f"\n[2/4] Pre-training on demos for {PRETRAIN_STEPS} iterations...")
    for i in range(PRETRAIN_STEPS):
        batch = agent.replay_buffer.sample_batch(BATCH_SIZE)
        agent.update(data=batch, update_timestep=i)
        if (i + 1) % 2000 == 0:
            print(f"  Iteration {i + 1}/{PRETRAIN_STEPS}")

    # --- STEP 3: ONLINE INTERACTION ---
    print(f"\n[3/4] Starting Online Learning (Max Steps: {MAX_STEPS})...")
    total_steps = 0
    steps_log, success_log = [], []

    while total_steps < MAX_STEPS:
        obs_dict, _ = agent.env.reset()
        obs = reconstruct_state(obs_dict)

        episode_data = []
        done = False

        while not done:
            # AWAC uses a stochastic policy for exploration
            action = agent.get_action(obs, deterministic=False)
            next_obs_dict, _, terminated, truncated, info = agent.env.step(action)
            next_obs = reconstruct_state(next_obs_dict)

            episode_data.append((obs, action, next_obs, terminated or truncated, info))
            obs = next_obs
            done = terminated or truncated
            total_steps += 1

        # Process episode for the buffer
        T_ep = len(episode_data)
        states_seq = [step[0] for step in episode_data] + [episode_data[-1][2]]
        dists, arm_dists = compute_distances(states_seq, T_ep)
        success = episode_data[-1][4].get("is_success", False)

        for t in range(T_ep):
            s, a, ns, d, _ = episode_data[t]
            reward = comp_reward(t, dists, arm_dists, weights, success, T_ep)
            agent.replay_buffer.store(s, a, reward, ns, float(d))

        # Perform gradient updates
        if agent.replay_buffer.size > BATCH_SIZE:
            # Update once per step taken in the environment (standard RL ratio)
            for _ in range(T_ep):
                batch = agent.replay_buffer.sample_batch(BATCH_SIZE)
                agent.update(data=batch, update_timestep=total_steps)

        # --- STEP 4: EVALUATION & VISUALIZATION ---
        if total_steps % EVAL_EVERY < T_ep:
            eval_successes = 0
            video_frames = []

            for trial in range(10):
                o_d, _ = eval_env.reset()
                o = reconstruct_state(o_d)
                d_eval = False

                while not d_eval:
                    if trial == 0 and SAVE_VIDEO:
                        video_frames.append(capture_frame(eval_env))

                    # Deterministic policy for evaluation
                    a = agent.get_action(o, deterministic=True)
                    o_d, _, term, trunc, inf = eval_env.step(a)
                    o = reconstruct_state(o_d)
                    d_eval = term or trunc

                if inf.get("is_success", False):
                    eval_successes += 1

            avg_sr = eval_successes / 10
            steps_log.append(total_steps)
            success_log.append(avg_sr)

            print(f"  Step: {total_steps} | Success Rate: {avg_sr:.2f}")

            if SAVE_VIDEO and video_frames:
                video_path = clips_dir / f"eval_step_{total_steps}.mp4"
                imageio.mimsave(video_path, video_frames, fps=30, codec="libx264")

            # Save periodic checkpoints
            torch.save(agent.ac.state_dict(), output_dir / f"policy_step_{total_steps}.pt")

    # Final Results
    plt.figure(figsize=(8, 5))
    plt.plot(steps_log, success_log, marker='o', linestyle='-', color='b')
    plt.title("AWAC Learning Progress (Pick and Place)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.savefig(output_dir / "learning_curve.png")
    print(f"\nTraining Complete. Plot saved to {output_dir}/learning_curve.png")


if __name__ == "__main__":
    main()
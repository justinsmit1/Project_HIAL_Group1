import time
import imageio
import numpy as np
import torch
import gymnasium as gym

from envs.task_envs import PnPNewRobotEnv
from utils.env_wrappers import (
    ActionNormalizer,
    ResetWrapper,
    TimeLimitWrapper,
    reconstruct_state,
)
from alg.awac import AWAC


# ======================================
# ENV SETUP
# ======================================


def make_env(render=False):
    env = PnPNewRobotEnv(render=render)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
    return env
# ======================================
# SAFE FRAME CAPTURE
# ======================================
def capture_frame(env):
    try:
        base = getattr(env, "unwrapped", env)
        pc = base.sim.physics_client

        view_matrix = pc.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=1.2,
            yaw=45,
            pitch=-35,
            roll=0,
            upAxisIndex=2,
        )

        proj_matrix = pc.computeProjectionMatrixFOV(
            fov=60, aspect=1.33, nearVal=0.01, farVal=10.0
        )

        _, _, px, _, _ = pc.getCameraImage(
            256, 256, view_matrix, proj_matrix,
            renderer=pc.ER_BULLET_HARDWARE_OPENGL
        )

        frame = np.reshape(px, (256, 256, 4))[:, :, :3]
        return frame.astype(np.uint8)

    except Exception as e:
        print(f"[WARN] Frame capture failed: {e}")
        return np.zeros((480, 640, 3), dtype=np.uint8)


# ======================================
# LOAD POLICY (SAFE)
# ======================================
def load_policy(policy_path):
    agent = AWAC(env_fn=lambda: make_env(render=False))

    state_dict = torch.load(policy_path, map_location="cpu")
    agent.ac.load_state_dict(state_dict)

    agent.ac.eval()
    return agent


# ======================================
# SAFE STEP WRAPPER
# ======================================
def safe_step(env, action):
    out = env.step(action)

    # Handle both Gym and Gymnasium formats
    if len(out) == 5:
        obs, reward, term, trunc, info = out
    else:
        obs, reward, done, info = out
        term, trunc = done, False

    if info is None:
        info = {}

    done = term or trunc
    return obs, reward, done, info


# ======================================
# VISUALIZATION
# ======================================
def visualize(policy_path, num_trajectories=5, save_video=True):
    print("\n=== Loading policy ===")
    agent = load_policy(policy_path)

    env = make_env(render=True)

    for traj in range(num_trajectories):
        print(f"\n=== Trajectory {traj + 1} ===")

        obs_dict, _ = env.reset()

        if obs_dict is None:
            print("[ERROR] reset returned None")
            continue

        obs = reconstruct_state(obs_dict)

        done = False
        frames = []
        steps = 0

        while not done:
            try:
                env.render()
                time.sleep(0.02)

                action = agent.get_action(obs, deterministic=True)

                next_obs_dict, _, done, info = safe_step(env, action)

                if next_obs_dict is None:
                    print("[WARN] next_obs_dict is None, ending episode")
                    break

                next_obs = reconstruct_state(next_obs_dict)

                # Save frame
                frames.append(capture_frame(env))

                obs = next_obs
                steps += 1

            except Exception as e:
                print(f"[ERROR] Step failed: {e}")
                break

        success = info.get("is_success", False)
        print(f"Steps: {steps} | Success: {success}")

        # Save video
        if save_video and len(frames) > 0:
            filename = f"trajectory_{traj + 1}.mp4"
            try:
                imageio.mimsave(filename, frames, fps=30)
                print(f"Saved video: {filename}")
            except Exception as e:
                print(f"[WARN] Could not save video: {e}")

    env.close()



# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    policy_path = "/Users/justinsmit_studie/Documents/project_hial/HIAL-PROJECT/saved/policy_learning/policy_latest.pt"  # <-- CHANGE THIS
    visualize(policy_path, num_trajectories=5)
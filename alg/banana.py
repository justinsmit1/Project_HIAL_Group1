from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np

from envs.task_envs import PnPNewRobotEnv
from utils.demos import prepare_demo_pool
from utils.env_wrappers import (
    ActionNormalizer,
    ResetWrapper,
    TimeLimitWrapper,
    TrajectoryRecord,
)


def feature_function(
    traj_pairs: List[Tuple[Dict[str, np.ndarray], np.ndarray]],
) -> np.ndarray:
    if len(traj_pairs) == 0:
        return np.zeros(8, dtype=np.float32)

    obj_goal_dists = []
    gripper_obj_dists = []
    gripper_goal_dists = []
    success_steps = []

    for state, _action in traj_pairs:
        obs = state["observation"]
        goal = state["desired_goal"]

        # Positions
        ee_pos = obs[0:3]  # gripper position
        obj_pos = obs[7:10]  # object position

        # Distances
        obj_goal_dist = np.linalg.norm(obj_pos - goal)
        gripper_obj_dist = np.linalg.norm(ee_pos - obj_pos)
        gripper_goal_dist = np.linalg.norm(ee_pos - goal)

        obj_goal_dists.append(obj_goal_dist)
        gripper_obj_dists.append(gripper_obj_dist)
        gripper_goal_dists.append(gripper_goal_dist)

        # success threshold
        success_steps.append(float(obj_goal_dist < 0.17))

    obj_goal_dists = np.array(obj_goal_dists)
    gripper_obj_dists = np.array(gripper_obj_dists)
    gripper_goal_dists = np.array(gripper_goal_dists)
    success_steps = np.array(success_steps)

    features = np.array(
        [
            obj_goal_dists.mean(),  # avg object → goal distance
            obj_goal_dists[-1],  # final object → goal distance
            obj_goal_dists.min(),  # closest object → goal
            gripper_obj_dists.mean(),  # avg gripper → object distance
            gripper_obj_dists[-1],  # final gripper → object distance
            gripper_goal_dists[-1],  # final gripper → goal distance
            len(traj_pairs),  # trajectory length
            success_steps.mean(),  # fraction of successful steps
        ],
        dtype=np.float32,
    )

    return features


# def feature_function(traj_pairs: List[Tuple[Dict[str, np.ndarray], np.ndarray]]) -> np.ndarray:
#     """Compute a feature vector summarising a trajectory.
#
#     The designed features are:
#     ##########################
#           Your Features
#     ##########################
#
#     Args:
#         traj_pairs: A list of (state_dict, action) tuples produced by rollout
#             or random_rollout.  Each state_dict must contain the keys
#             "observation" (raw obs array, object xyz at indices 7–9) and
#             "desired_goal" (goal xyz).
#
#     Returns:
#         A float32 array of shape (d,).  Returns the zero vector for empty input.
#     """
#     if len(traj_pairs) == 0:
#         return np.zeros(5, dtype=np.float32)
#
#     dists = []
#     sum_dist_to_goal = []
#
#     for state, _action in traj_pairs:
#         obs = state["observation"]
#         achieved = state["achieved_goal"]
#         goal = state["desired_goal"]
#
#         # This computes the distance to the goal and this is done at each time step
#         obj_pos = obs[7:10]
#         dist = np.linalg.norm(obj_pos - goal)
#
#         #print(f"achieved vs goal: {dist} -> {goal}")
#
#         sum_dist_to_goal.append(float(dist < 0.17))  # threshold for success
#
#         dists.append(dist)
#
#     dists = np.array(dists)
#     sum_dist_to_goal = np.array(sum_dist_to_goal)
#
#     features = np.array(
#         [
#             dists.mean(), #Average distance of object to goal
#             dists[-1], #Final distance
#             dists.min(), #Closest the agent got to the goal
#             len(traj_pairs), # number of steps in trajectory
#             # distance hand to banana
#             # distance banana to goal
#             sum_dist_to_goal.mean()
#         ],
#         dtype=np.float32,
#     )
#
#     return features


def capture_frame(env: Any, width: int = 320, height: int = 240) -> np.ndarray:
    """Render the current simulation state to an image via PyBullet offscreen rendering.

    Attempts a hardware-accelerated render using ER_BULLET_HARDWARE_OPENGL.
    Falls back to a black frame of the requested dimensions if rendering fails

    Args:
        env: A gym environment whose .unwrapped.sim.physics_client
            exposes the PyBullet physics client.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        A uint8 array of shape (height, width, 3) in RGB channel order.
    """
    try:
        base = getattr(env, "unwrapped", env)
        pc = base.sim.physics_client

        view_matrix = pc.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 0.0],
            distance=1.0,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = pc.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(width) / float(height),
            nearVal=0.01,
            farVal=10.0,
        )

        _, _, px, _, _ = pc.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pc.ER_BULLET_HARDWARE_OPENGL,
        )
        rgba = np.reshape(px, (height, width, 4))
        return rgba[:, :, :3].astype(np.uint8, copy=False)

    except Exception:
        return np.zeros((height, width, 3), dtype=np.uint8)


def setup_environment(*, render: bool = False) -> Any:
    """Construct and initialise the Pick-and-Place environment with standard wrappers.

    Wrapper stack (inner → outer):
    PnPNewRobotEnv → ResetWrapper → ActionNormalizer → TimeLimitWrapper (150 steps).

    The environment is seeded with seed=0 immediately after construction.

    Args:
        render: If True, opens a PyBullet GUI window.

    Returns:
        The fully wrapped, reset environment.
    """

    env = PnPNewRobotEnv(render=render)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    env.reset(seed=0)

    return env


def rollout(
    env: Any,
    action_seq: np.ndarray,
    *,
    options: Optional[Dict[str, Any]] = None,
    max_steps: int = 150,
) -> Tuple[List[Tuple[Dict[str, np.ndarray], np.ndarray]], List[np.ndarray]]:
    """Execute a fixed action sequence in the environment and record the trajectory.

    Args:
        env: Wrapped gym environment (see setup_environment).
        action_seq: Array of shape (T, action_dim) containing the pre-recorded
            actions to replay.
        options: Optional reset options forwarded to env.reset.
        max_steps: Hard cap on the number of steps executed, regardless of
            action_seq length.

    Returns:
        A tuple (traj_pairs, frames) where:
            * traj_pairs is a list of (state dict, action) pairs;
            * frames is a list of uint8 RGB arrays captured after each step.
    """
    T = min(int(action_seq.shape[0]), int(max_steps))
    frames: List[np.ndarray] = []
    traj_pairs: List[Tuple[Dict[str, np.ndarray], np.ndarray]] = []

    state, _info = env.reset(seed=0, options=options)
    frames.append(capture_frame(env))

    for t in range(T):
        action = action_seq[t]

        next_state, reward, terminated, truncated, info = env.step(action)

        traj_pairs.append((state, action))
        frames.append(capture_frame(env))

        state = next_state

        if terminated or truncated:
            break

    return traj_pairs, frames


def random_rollout(
    env: Any,
    *,
    max_steps: int = 150,
) -> Tuple[List[Tuple[Dict[str, np.ndarray], np.ndarray]], List[np.ndarray]]:
    """Execute a random policy in the environment and record the trajectory.

    Actions are sampled uniformly from env.action_space at every step.

    Args:
        env: Wrapped gym environment (see :setup_environment).
        max_steps: Maximum number of environment steps to execute.

    Returns:
        A tuple (traj_pairs, frames) where:
            * traj_pairs is a list of (state dict, action) pairs;
            * frames is a list of uint8 RGB arrays.
    """
    frames: List[np.ndarray] = []
    traj_pairs: List[Tuple[Dict[str, np.ndarray], np.ndarray]] = []

    state, _info = env.reset(seed=0)
    frames.append(capture_frame(env))

    for _ in range(max_steps):

        action = env.action_space.sample()

        next_state, reward, terminated, truncated, info = env.step(action)

        traj_pairs.append((state, action))
        frames.append(capture_frame(env))

        state = next_state

        if terminated or truncated:
            break

    return traj_pairs, frames


def main() -> None:
    """generate expert and random trajectory clips, then serialise records.

    1. Load all expert demos from ``<repo_root>/demo_data/PickAndPlace``.
    2. Roll out each demo in the environment, saving an MP4 clip and computing
       feature vectors.
    3. Generate 10 additional random-policy clips.
    4. Serialise all TrajectoryRecord objects to
       ``<repo_root>/saved/trajectory_records.json``.
    """
    env = setup_environment(render=True)  # CHECK

    repo_root = Path(__file__).resolve().parents[1]
    demo_dir = repo_root / "demo_data" / "PickAndPlace"
    saved_dir = repo_root / "saved"
    clips_dir = saved_dir / "clips"
    saved_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    demos = prepare_demo_pool(demo_dir, verbose=True)
    print(f"\nLoaded {len(demos)} expert demos from: {demo_dir}")

    saved_records: List[TrajectoryRecord] = []
    import csv

    feature_rows = []

    fps = 30
    writer_kwargs: Dict[str, Any] = dict(
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-preset", "ultrafast", "-crf", "28"],
    )

    print(f"\nGenerating {len(demos)} expert clips")

    for i, demo in enumerate(demos):
        action_seq = demo["action_trajectory"]
        traj_pairs, frames = rollout(env, action_seq)

        clip_path = clips_dir / f"expert_{i}.mp4"

        imageio.mimsave(clip_path, frames, **writer_kwargs)

        features = feature_function(traj_pairs)

        feature_rows.append(
            {
                "type": "expert",
                "clip": str(clip_path),
                **{f"f{j}": float(features[j]) for j in range(len(features))},
            }
        )

        record = TrajectoryRecord(
            clip_path=str(clip_path),
            features=features,
        )

        saved_records.append(record)

    print(f"\nGenerating 10 random clips")
    for i in range(10):
        traj_pairs, frames = random_rollout(env)

        clip_path = clips_dir / f"random_{i}.mp4"

        imageio.mimsave(clip_path, frames, **writer_kwargs)

        features = feature_function(traj_pairs)

        feature_rows.append(
            {
                "type": "random",
                "clip": str(clip_path),
                **{f"f{j}": float(features[j]) for j in range(len(features))},
            }
        )

        record = TrajectoryRecord(
            clip_path=str(clip_path),
            features=features,
        )

        saved_records.append(record)

    env.close()

    out_path = saved_dir / "trajectory_records.json"
    with open(out_path, "w") as f:
        json.dump([r.to_json() for r in saved_records], f, indent=2)

    print(f"Saved {len(saved_records)} trajectory records to {out_path}")

    csv_path = out_path / "trajectory_features.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=feature_rows[0].keys())
        writer.writeheader()
        writer.writerows(feature_rows)

    print(f"Saved feature CSV to {csv_path}")


if __name__ == "__main__":
    main()

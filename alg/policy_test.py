import torch
import numpy as np
from pathlib import Path
from envs.task_envs import PnPNewRobotEnv
from utils.env_wrappers import (
    ActionNormalizer,
    ResetWrapper,
    TimeLimitWrapper,
    reconstruct_state,
)
from alg.awac import AWAC, ReplayBuffer
from alg.policy_learn import make_env, FlattenObsWrapper


def load_final_policy(path_to_saved_policy):
    """
    load your final trained policy

    Args:
        path_to_saved_policy (str): the path to your saved policy model

    Returns:
        your saved policy model under the corresponding path
    """
    agent = AWAC(env_fn=make_env)
    agent.ac.load_state_dict(torch.load(path_to_saved_policy))
    agent.ac.eval()  # set to evaluation mode
    return agent


def get_policy_action(state, saved_policy_model):
    """
    get the action that the policy decides to take for the given environment state

    Args:
        state (dict): the state of the environment returned by the env.step() or env.reset(), which is a dictionary including keys of "observation", "achieved_goal", and "desired_goal"
        saved_policy_model: a saved model in the same format as the one returned by your load_final_policy() function

    Returns:
        action (np.array): the action that the saved policy model decides to take under the given state
    """
    flat_state = reconstruct_state(state)
    action = saved_policy_model.get_action(flat_state, deterministic=True)
    return action

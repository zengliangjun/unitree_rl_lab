
from isaaclab_rl import rsl_rl

import gymnasium as gym
import torch

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv



class RslRlVecEnvWrapper(rsl_rl.RslRlVecEnvWrapper):

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        super(RslRlVecEnvWrapper, self).__init__(env, clip_actions)

        if hasattr(self.unwrapped, "action_manager"):
            if 1 != len(self.unwrapped.action_manager.active_terms):
                self.policy_dim_actions = {}
                for name, dim in zip(self.unwrapped.action_manager.active_terms, self.unwrapped.action_manager.action_term_dim):
                    self.policy_dim_actions[name] = dim

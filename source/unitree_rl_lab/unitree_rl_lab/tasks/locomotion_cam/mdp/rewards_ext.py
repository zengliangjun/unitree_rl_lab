from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, RewardTermCfg

from unitree_rl_lab.tasks.locomotion.mdp import commands

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class action_rate_l2_withname(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        action_name = cfg.params.get("action_name", None)
        assert action_name is not None, "action_name must be specified in the reward term config."

        start_dim = 0
        end_dim = 0
        for name, dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
            start_dim = end_dim
            end_dim += dim
            if name != action_name:
                continue

        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, env: ManagerBasedRLEnv,
                action_name: str) -> torch.Tensor:

        error = torch.square(env.action_manager.action - env.action_manager.prev_action)[:, self.start_dim: self.end_dim]
        return torch.sum(error, dim=1)

class action_rate_l2_withname_ext(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        action_name = cfg.params.get("action_name", None)
        assert action_name is not None, "action_name must be specified in the reward term config."

        start_dim = 0
        end_dim = 0
        for name, dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
            start_dim = end_dim
            end_dim += dim
            if name != action_name:
                continue

        self.start_dim = start_dim
        self.end_dim = end_dim

        self.prev_prev_action = torch.zeros(env.num_envs, self.end_dim - self.start_dim, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.prev_prev_action.zero_()
        else:
            self.prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv,
                action_name: str) -> torch.Tensor:

        error = (env.action_manager.action - 2 * env.action_manager.prev_action)[:, self.start_dim: self.end_dim] + self.prev_prev_action
        error = torch.square(error)

        self.prev_prev_action[...] = env.action_manager.prev_action[:, self.start_dim: self.end_dim]
        return torch.sum(error, dim=1)

def is_alive_exp(env: ManagerBasedRLEnv, command_name: str, std: float = 0.05) -> torch.Tensor:
    """Reward for being alive."""
    command: commands.CommandWithEpisodeInfo = env.command_manager.get_term(command_name)
    aliving = (command.average_episode_length / env.max_episode_length) / std

    return (~env.termination_manager.terminated).float() * math.exp(- aliving)

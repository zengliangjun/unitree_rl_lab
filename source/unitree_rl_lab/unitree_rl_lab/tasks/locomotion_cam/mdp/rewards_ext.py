from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, RewardTermCfg

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

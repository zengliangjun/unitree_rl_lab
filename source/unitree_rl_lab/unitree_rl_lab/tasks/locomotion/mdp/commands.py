from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import Sequence, TYPE_CHECKING

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class CommandWithEpisodeInfo(UniformVelocityCommand):

    cfg: UniformLevelVelocityCommandCfg

    average_episode_length: float
    episode_length_buffer: torch.Tensor

    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.average_episode_length = 0
        self.episode_length_buffer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            self.average_episode_length = 0
            self.episode_length_buffer[:] = 0
        else:
            self.average_episode_length = self.average_episode_length * 0.9 + torch.mean(self.episode_length_buffer[env_ids].float()).cpu().item() * 0.1
            self.episode_length_buffer[env_ids] = 0

        return super().reset(env_ids)

    def compute(self, dt: float):
        self.episode_length_buffer[:] += 1
        super().compute(dt)



@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = CommandWithEpisodeInfo

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import Sequence, TYPE_CHECKING

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class StompCommand(UniformVelocityCommand):

    cfg: StompCommandCfg

    average_episode_length: float
    episode_length_buffer: torch.Tensor

    '''
    no vel z, now it is flag for Stomp

    '''

    def __init__(self, cfg: StompCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.average_episode_length = 0
        self.episode_length_buffer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.metrics.pop("error_vel_yaw")

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )


    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            self.average_episode_length = 0
            self.episode_length_buffer[:] = 0
        else:
            self.average_episode_length = self.average_episode_length * 0.9 + torch.mean(self.episode_length_buffer[env_ids].float()).cpu().item() * 0.1
            self.episode_length_buffer[env_ids] = 0

        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids, 2] = 1
        self.vel_command_b[self.is_standing_env, 2] = 0

    def _update_command(self):
        super()._update_command()
        self.vel_command_b[:, 2] = 1
        self.vel_command_b[self.is_standing_env, 2] = 0

    def compute(self, dt: float):
        self.episode_length_buffer[:] += 1
        super().compute(dt)


@configclass
class StompCommandCfg(UniformVelocityCommandCfg):
    class_type: type = StompCommand



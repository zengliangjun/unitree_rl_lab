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

        self.feet_global_phases = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.feet_swing_phases = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        global_phase = self.feet_global_phases
        swing_phase = self.feet_swing_phases

        sin_phase = torch.sin(global_phase * torch.pi * 2.0)
        cos_phase = torch.cos(global_phase * torch.pi * 2.0)

        swing_sin_phase = torch.sin(swing_phase * torch.pi)
        swing_cos_phase = torch.cos(swing_phase * torch.pi)

        return torch.cat([sin_phase, cos_phase, swing_sin_phase, swing_cos_phase], dim=-1)

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
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
        self.vel_command_b[env_ids, :] = 1
        self.vel_command_b[self.is_standing_env, :] = 0

    def _update_command(self):
        super()._update_command()
        self.vel_command_b[:, :] = 1
        self.vel_command_b[self.is_standing_env, :] = 0

        ##
        global_phase = ((self.episode_length_buffer * self._env.step_dt) % self.cfg.period / self.cfg.period).unsqueeze(1)
        phases = []
        for offset_ in self.cfg.offset:
            phase = (global_phase + offset_) % 1.0
            phases.append(phase)

        phases = torch.cat(phases, dim=-1)
        phases[self.is_standing_env] = 0

        self.feet_global_phases[...] = phases #   * torch.pi * 2
        ##
        swing_phase = torch.clamp_min(phases - self.cfg.threshold, min=0.0) / (1 - self.cfg.threshold)
        swing_phase[self.is_standing_env] = 0

        self.feet_swing_phases[...] = swing_phase #  * torch.pi


    def compute(self, dt: float):
        self.episode_length_buffer[:] += 1
        super().compute(dt)


@configclass
class StompCommandCfg(UniformVelocityCommandCfg):
    class_type: type = StompCommand

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING

    period: float = 0.8
    offset: tuple[float, float] = (0.0, 0.5)
    threshold: float = 0.55

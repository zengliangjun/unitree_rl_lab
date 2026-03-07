from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import Sequence, TYPE_CHECKING

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class CommandWithPhase(UniformVelocityCommand):

    cfg: CommandWithPhaseCfg

    average_episode_length: float
    episode_length_buffer: torch.Tensor

    def __init__(self, cfg: CommandWithPhaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.average_episode_length = 0
        self.episode_length_buffer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.feet_global_phases = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.feet_swing_phases = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)

    @property
    def feet_phases(self) -> torch.Tensor:

        global_phase = self.feet_global_phases
        swing_phase = self.feet_swing_phases

        sin_phase = torch.sin(global_phase * torch.pi * 2.0)
        cos_phase = torch.cos(global_phase * torch.pi * 2.0)

        swing_sin_phase = torch.sin(swing_phase * torch.pi)
        swing_cos_phase = torch.cos(swing_phase * torch.pi)

        return torch.cat([sin_phase, cos_phase, swing_sin_phase, swing_cos_phase], dim=-1)


    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            self.average_episode_length = 0
            self.episode_length_buffer[:] = 0
        else:
            self.average_episode_length = self.average_episode_length * 0.9 + torch.mean(self.episode_length_buffer[env_ids].float()).cpu().item() * 0.1
            self.episode_length_buffer[env_ids] = 0

        return super().reset(env_ids)

    def _update_command(self):
        super()._update_command()

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
        swing_phases = torch.clamp_min(phases - self.cfg.threshold, min=0.0) / (1 - self.cfg.threshold)
        swing_phases[self.is_standing_env] = 0

        self.feet_swing_phases[...] = swing_phases #  * torch.pi

    def compute(self, dt: float):
        self.episode_length_buffer[:] += 1
        super().compute(dt)


@configclass
class CommandWithPhaseCfg(UniformVelocityCommandCfg):
    class_type: type = CommandWithPhase

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING

    period: float = 0.8
    offset: tuple[float, float] = (0.0, 0.5)
    threshold: float = 0.55


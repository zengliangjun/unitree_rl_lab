
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.managers import CommandTerm

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

from loguru import logger as ulogger

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .command_squat_cfg import SuqatCommandCfg

"""
  + pi / 2
  |
  |
  +-------- 0
  |
  |
  - pi / 2
"""

class SuqatCommand(CommandTerm):

    cfg: SuqatCommandCfg
    def __init__(self, cfg: SuqatCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset: Articulation = env.scene[cfg.asset_cfg.name]
        cfg.asset_cfg.resolve(env.scene)

        self.suqat_phase = torch.zeros(self.num_envs, 1, device=self.device)
        self.suqat_command_phase = torch.zeros(self.num_envs, 1, device=self.device)
        self.suqat_phase_vel = torch.zeros(self.num_envs, 1, device=self.device)
        self.is_init_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_max_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_finished_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buffer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.mean_episode_length = 0


        pos_limits = self.asset.data.joint_pos_limits[:, self.cfg.asset_cfg.joint_ids].clone()  # n * j * 2
        self.cpos = torch.mean(torch.mean(pos_limits, dim = -1), dim = -1, keepdim=True)  # n * j
        self.rad = torch.mean((pos_limits[:, :, 1] - pos_limits[:, :, 0]) / 2, dim = -1, keepdim=True)  # n * j

        pos = torch.mean(self.asset.data.default_joint_pos[:, self.cfg.asset_cfg.joint_ids], dim = -1, keepdim=True)  # n * 1
        self.init_phase = torch.asin((self.cpos - pos) / self.rad)

        self.metrics["error_knee_pos"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "SuqatCommand:\n"
        msg += f"\tTarget dimension: {tuple(self.suqat_phase.shape[1:])}\n"
        msg += f"\tCommand dimension: {tuple(self.suqat_command_phase.shape[1:])}\n"
        msg += f"\tvel dimension: {tuple(self.suqat_phase_vel.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        com = torch.cat((self.suqat_command_phase, self.suqat_phase), dim=-1)
        com_cos = torch.cos(com)
        com_sin = torch.sin(com)
        return torch.cat((com / torch.pi, com_cos, com_sin, self.suqat_phase_vel), dim=-1)

    @property
    def command_pos(self) -> torch.Tensor:
        pos = self.cpos - torch.sin(self.suqat_phase) * self.rad
        return pos

    """

    for compute resample

    """
    def compute(self, dt: float):
        """Compute the command.

        Args:
            dt: The time step passed since the last call to compute.
        """
        # update the metrics based on current state
        self._update_metrics()
        # reduce the time left before resampling
        self.time_left -= dt
        # resample the command if necessary
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample_compute(resample_env_ids)
        # update the command
        self._update_command()


    def _resample_compute(self, env_ids: Sequence[int]):
        """Resample the command.

        This function resamples the command and time for which the command is applied for the
        specified environment indices.

        Args:
            env_ids: The list of environment IDs to resample.
        """
        if len(env_ids) != 0:
            # resample the time left before resampling
            self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)

            # clean flags
            self.is_init_env[env_ids] = 0
            self.is_max_env[env_ids] = 0
            self.is_finished_flags[env_ids] = 0
            # resample the command
            self._resample_command_compute(env_ids)
            # increment the command counter
            self.command_counter[env_ids] += 1

    def _resample_command_compute(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction

        full_times = r.uniform_(*self.cfg.ranges.full_times)
        self.suqat_phase_vel[env_ids, 0] = torch.pi / full_times

        ## update suqat command phase
        self.suqat_command_phase[env_ids, 0] = r.uniform_(*self.cfg.ranges.suqat_phase)

        uniform_prop = r.uniform_(0.0, 1.0)
        self.is_init_env[env_ids] = uniform_prop <= self.cfg.rel_compute_init_envs
        self.is_max_env[env_ids] = torch.logical_and(uniform_prop > self.cfg.rel_compute_init_envs,
                uniform_prop <= self.cfg.rel_compute_init_envs + self.cfg.rel_compute_max_envs)

        self.suqat_command_phase[self.is_init_env] = self.init_phase[self.is_init_env]
        self.suqat_command_phase[self.is_max_env] = self.cfg.ranges.suqat_phase[0]     ##  suqat to min pos

        ## get pos
        pos = self.asset.data.joint_pos[:, self.cfg.asset_cfg.joint_ids]
        phase = torch.asin((self.cpos[env_ids, :] - pos[env_ids, :]) / self.rad[env_ids, :])
        self.suqat_phase[env_ids, :] = phase.mean(dim = -1, keepdim=True)

        #  debug_string = f"command_compute ({self.suqat_command_phase[0, 0]})  phase({self.suqat_phase[0, 0]})  vel ({self.suqat_phase_vel[0, 0]})"
        self.suqat_phase_vel[env_ids, :] *= torch.sign(self.suqat_command_phase[env_ids] - self.suqat_phase[env_ids])
        #  debug_string += f" vel({self.suqat_phase_vel[0, 0]}) "
        #  ulogger.info(debug_string)


    """

    for reset resample

    """
    def _resample_command(self, env_ids: Sequence[int]):
        self.mean_episode_length = self.mean_episode_length * 0.9 + torch.mean(self.episode_length_buffer[env_ids].float()).cpu().item() * 0.1
        self.episode_length_buffer[env_ids] = 0
        """
        for reset env_ids
        """
        self.is_init_env[env_ids] = 0
        self.is_max_env[env_ids] = 0
        self.is_finished_flags[env_ids] = 0

        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction

        full_times = r.uniform_(*self.cfg.ranges.full_times)
        self.suqat_phase_vel[env_ids, 0] = torch.pi / full_times

        ## update suqat command phase
        self.suqat_command_phase[env_ids, 0] = r.uniform_(*self.cfg.ranges.suqat_phase)
        self.is_init_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_reset_init_envs
        self.suqat_command_phase[self.is_init_env] = self.init_phase[self.is_init_env]

        ## get pos
        pos = self.asset.data.joint_pos[:, self.cfg.asset_cfg.joint_ids]
        phase = torch.asin((self.cpos[env_ids, :] - pos[env_ids, :]) / self.rad[env_ids, :])
        self.suqat_phase[env_ids, :] = phase.mean(dim = -1, keepdim=True)

        #  debug_string = f"command ({self.suqat_command_phase[0, 0]})  phase({self.suqat_phase[0, 0]})  vel ({self.suqat_phase_vel[0, 0]})"

        self.suqat_phase_vel[env_ids, :] *= torch.sign(self.suqat_command_phase[env_ids] - self.suqat_phase[env_ids])
        #  debug_string += f" vel({self.suqat_phase_vel[0, 0]}) "
        #  ulogger.info(debug_string)

    def _update_command(self):
        self.episode_length_buffer += 1
        #  debug_string = f"update command ({self.suqat_command_phase[0, 0]})  phase({self.suqat_phase[0, 0]})  vel ({self.suqat_phase_vel[0, 0]})"
        self.suqat_phase += self.suqat_phase_vel * self._env.step_dt
        #  debug_string += f" phase2({self.suqat_phase[0, 0]}) "

        vel_flag = self.suqat_phase_vel >= 0
        if torch.sum(vel_flag) > 0:
            flag = vel_flag[:, 0]
            finished_flags = self.suqat_phase[flag] >= self.suqat_command_phase[flag]
            self.is_finished_flags[flag] = finished_flags[:, 0]
            self.suqat_phase[vel_flag] = torch.clamp_max(self.suqat_phase[vel_flag], self.suqat_command_phase[vel_flag])

        vel_flag = self.suqat_phase_vel < 0
        if torch.sum(vel_flag) > 0:
            flag = vel_flag[:, 0]
            finished_flags = self.suqat_phase[flag] <= self.suqat_command_phase[flag]
            self.is_finished_flags[flag] = finished_flags[:, 0]
            self.suqat_phase[vel_flag] = torch.clamp_min(self.suqat_phase[vel_flag], self.suqat_command_phase[vel_flag])

        #  debug_string += f" phase3({self.suqat_phase[0, 0]}) "
        #  ulogger.info(debug_string)

    def _update_metrics(self):
        pos = self.asset.data.joint_pos[:, self.cfg.asset_cfg.joint_ids]
        # logs data
        self.metrics["error_knee_pos"] += (
            torch.norm(self.command_pos - pos, dim=-1)
        )

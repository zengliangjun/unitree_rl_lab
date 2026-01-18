
from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .commands import CommandWithEpisodeInfo

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_ext_cfg import CommandExtCfg

class CommandExt(CommandWithEpisodeInfo):

    cfg: CommandExtCfg

    def __init__(self, cfg: CommandExtCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg

        # self.is_standing_env
        self.is_walking_env = torch.zeros_like(self.is_standing_env)
        self.is_spining_env = torch.zeros_like(self.is_standing_env)
        self.is_walking_spining_env = torch.zeros_like(self.is_standing_env)

        self.probs = torch.tensor([self.cfg.rel_standing_envs,
                                   self.cfg.rel_walking_envs,
                                   self.cfg.rel_spining_envs,
                                   self.cfg.rel_walking_spining_envs], device = self.device)
        '''
        self.standing_probs = self.cfg.rel_standing_envs
        self.walking_probs = self.standing_probs + self.cfg.rel_walking_envs
        self.spining_probs = self.walking_probs + self.cfg.rel_spining_envs
        '''


    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "CommandExt:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}\n"
        msg += f"\tWalking probability: {self.cfg.rel_walking_envs}\n"
        msg += f"\tSpinning probability: {self.cfg.rel_spining_envs}\n"
        msg += f"\tWalking Spinning probability: {self.cfg.rel_walking_spining_envs}\n"
        return msg


    def _resample_command(self, env_ids: Sequence[int]):
        self.is_standing_env[env_ids] = False
        self.is_walking_env[env_ids] = False
        self.is_spining_env[env_ids] = False
        self.is_walking_spining_env[env_ids] = False

        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        indices = torch.multinomial(self.probs, num_samples=len(env_ids), replacement=True)

        torch_env_ids = torch.tensor(env_ids, device=self.device)

        for cls in range(self.probs.shape[0]):
            # 筛选出当前类别的所有样本索引，再提取对应数据
            cls_mask = (indices == cls)
            if 0 == cls:  # stand
                stand_ids = torch_env_ids[cls_mask]  #
                if 0 == len(stand_ids):
                    continue
                self.vel_command_b[stand_ids, :3] = 0.0
                self.is_standing_env[stand_ids] = True

            elif 1 == cls:  # walk
                walk_ids = torch_env_ids[cls_mask]  #
                if 0 == len(walk_ids):
                    continue
                self.vel_command_b[walk_ids, 2:] = 0.0
                self.is_walking_env[walk_ids] = True

            elif 2 == cls:  # spin
                spin_ids = torch_env_ids[cls_mask]  #
                if 0 == len(spin_ids):
                    continue
                self.vel_command_b[spin_ids, :2] = 0.0
                self.is_spining_env[spin_ids] = True

            elif 3 == cls:  # walk spin
                walk_spin_ids = torch_env_ids[cls_mask]  #
                if 0 == len(walk_spin_ids):
                    continue
                self.is_walking_spining_env[walk_spin_ids] = True

        stand_mask = torch.norm(self.vel_command_b[env_ids], dim=-1) < 0.1
        stand_ids = torch_env_ids[stand_mask]  #
        if 0 != len(stand_ids):
            self.vel_command_b[stand_ids] = 0
            self.is_standing_env[stand_ids] = True
            self.is_walking_env[stand_ids] = False
            self.is_spining_env[stand_ids] = False
            self.is_walking_spining_env[stand_ids] = False

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from collections.abc import Sequence

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

class BodiesSymmetry(ManagerTermBase):
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = self._env.scene[asset_cfg.name]
        self.asset_cfg = asset_cfg
        self.command_name = cfg.params["command_name"]

        self._init_buffers()
        # 初始化标志位

    def _init_buffers(self):
        # 初始化足接触统计的均值与方差缓冲区 (一维数据)
        posecount = len(self.asset_cfg.body_ids)
        self.episode_variance_buf = torch.zeros((self.num_envs, posecount, 3),
                              device=self.device, dtype=torch.float)
        self.episode_mean_buf = torch.zeros_like(self.episode_variance_buf)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # 重置足接触统计的缓冲区，并导出导数数据
        if env_ids is None or len(env_ids) == 0:
            return

        # 清空所有足接触统计缓冲区
        buffers = [
            self.episode_variance_buf,
            self.episode_mean_buf,
        ]

        for buf in buffers:
            buf[env_ids] = 0

    def _update_flag(self):
        command: UniformVelocityCommand = self._env.command_manager.get_term(self.command_name)
        self.stand_flag = torch.logical_or(command.is_standing_env ,
                                           self._env.episode_length_buf <= 1)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        # 利用增量更新方法计算当前episode的均值和方差
        episode_length_buf = self._env.episode_length_buf

        # 计算均值：根据新差值delta0更新均值缓冲区
        delta0 = diff - self.episode_mean_buf
        self.episode_mean_buf += delta0 / episode_length_buf[:, None, None]

        # 计算方差：利用delta0和新均值计算更新方差缓冲区
        delta1 = diff - self.episode_mean_buf
        self.episode_variance_buf = (
            self.episode_variance_buf * (episode_length_buf[:, None, None] - 2)
            + delta0 * delta1
        ) / (episode_length_buf[:, None, None] - 1)

        # 当episode刚开始时重置方差，防止数值异常
        new_episode_mask = episode_length_buf <= 1
        # self.episode_mean_buf[new_episode_mask] = 0
        self.episode_variance_buf[new_episode_mask] = 0

    def _calcute_pose(self):
        pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids] - self.asset.data.root_pos_w[:, None, :]

        # 重复根链接的旋转四元数，使其与 pos 的维度匹配
        # 即将根链接的四元数应用到每个 body 上
        quat_w = torch.repeat_interleave(self.asset.data.root_quat_w[:, None, :], pos_w.shape[1], dim=1)
        # 对提取的位置进行逆旋转转换，将世界坐标系位置转换到机器人基座坐标系
        try:
            pos_b = math_utils.quat_apply_inverse(quat_w, pos_w)
        except:
            pos_b = math_utils.quat_rotate_inverse(quat_w, pos_w)

        self._calculate_episode(pos_b)

    def _calculate_meanx(self, error_std: float = 0.06):
        episode_mean = self.episode_mean_buf
        mean0 = episode_mean[:, ::2, :]
        mean1 = episode_mean[:, 1::2, :]

        meanxz = torch.abs(mean0[..., 0] - mean1[..., 0])
        rewmeanx = torch.exp(- meanxz / error_std)
        rewmeanx = torch.sum(rewmeanx, dim = -1)
        return rewmeanx

    def _calculate_varx(self, std_ranges: list[float] = [0.08, 0.21], error_std: float = 0.06):

        episode_variance = self.episode_variance_buf
        episode_std = torch.sqrt(episode_variance)

        # walk
        stdx = torch.abs(torch.clamp_max(episode_std[..., 0] - std_ranges[0], 0)) + \
                        torch.clamp_min(episode_std[..., 0] - std_ranges[1], 0)
        rew_walkstdx = torch.exp(- stdx / error_std)
        rew_walkstdx = torch.sum(rew_walkstdx, dim = -1) / 2
        # stand
        rew_standstdx = torch.exp(- episode_std[..., 0] / error_std)
        rew_standstdx = torch.sum(rew_standstdx, dim = -1) / 2

        stand_flag = self.stand_flag
        rew_walkstdx[stand_flag] = 0

        walkflag = torch.logical_not(self.stand_flag)
        rew_standstdx[walkflag] = 0

        return rew_standstdx + rew_walkstdx


    def __call__(self,
            env: ManagerBasedRLEnv,
            command_name: str = "base_velocity",
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
            std_ranges: list[float] = [0.08, 0.21],
            error_std: float = 0.06
        ) -> torch.Tensor:
        self._update_flag()
        self._calcute_pose()
        return self._calculate_meanx(error_std) + self._calculate_varx(std_ranges, error_std)


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ZMP(ManagerTermBase):

    _env: ManagerBasedRLEnv

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        self.com_vel_w_history = torch.zeros((env.num_envs, 6), device=env.device)
        self.com_acc_history = torch.zeros((env.num_envs, 6), device=env.device)
        self.zmp_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
        self.dt = env.step_dt

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.com_vel_w_history.zero_()
            self.com_acc_history.zero_()
        else:
            self.com_vel_w_history[env_ids] = 0.0
            self.com_acc_history[env_ids] = 0.0

    def compute_zmp(self):
        com_pos = self.asset.data.root_com_pos_w  # 根关节位置

        com_acc = (self.asset.data.root_com_vel_w - self.com_vel_w_history) / self._env.step_dt
        com_acc = self.com_acc_history * 0.8 + com_acc * 0.2

        self.zmp_pos_w[...] = com_pos - (com_pos / 9.81) * com_acc[:, :3]

        self.com_vel_w_history[...] = self.asset.data.root_com_vel_w
        self.com_acc_history[...] = com_acc

    def __call__(self,
            env: ManagerBasedRLEnv,
            std: float,
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

        self.compute_zmp()

        pos = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids]
        quat_w = torch.repeat_interleave(self.asset.data.root_link_quat_w[:, None, :], pos.shape[1], dim=1)

        try:
            pos_b = math_utils.quat_apply_inverse(quat_w, pos)
            pos = torch.mean(pos_b[:, :, :2], dim=1)
            zmp_b = math_utils.quat_apply_inverse(self.asset.data.root_link_quat_w, self.zmp_pos_w)
        except:
            pos_b = math_utils.quat_rotate_inverse(quat_w, pos)
            pos = torch.mean(pos_b[:, :, :2], dim=1)
            zmp_b = math_utils.quat_rotate_inverse(self.asset.data.root_link_quat_w, self.zmp_pos_w)

        left_ankle = pos_b[:, 0, :2]
        right_ankle = pos_b[:, 1, :2]

        support_vec = right_ankle - left_ankle
        support_len = torch.linalg.norm(support_vec, dim=1)
        support_len = torch.clamp(support_len, min=0.05)

        support_dir = support_vec / (support_len.unsqueeze(1) + 1e-6)
        normal_vec = torch.stack([-support_dir[:, 1], support_dir[:, 0]], dim=1)

        zmp_proj = zmp_b[:, :2] - left_ankle
        zmp_dist = torch.sum(zmp_proj * normal_vec, dim=1)
        zmp_dist = torch.clamp(zmp_dist, min=-0.5, max=0.5)

        zmp_along = torch.sum(zmp_proj * support_dir, dim=1)

        margin = support_len * 0.1
        # stable = (com_along >= -margin) & (com_along <= (support_len + margin))

        dist_reward = torch.exp(- torch.abs(zmp_dist) / std)
        # range_reward = torch.where(stable, 1.0, 0.2)
        range_reward = torch.sigmoid(
                    (zmp_along + margin) / (0.1 + support_len)
                ) * torch.sigmoid(
                    (support_len + margin - zmp_along) / (0.1 + support_len)
                )

        total_reward = dist_reward * range_reward
        return total_reward

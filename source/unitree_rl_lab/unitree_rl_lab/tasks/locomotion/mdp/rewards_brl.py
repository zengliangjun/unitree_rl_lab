from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class action_smoothness1(ManagerTermBase):

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

        action = env.action_manager.action[:, self.start_dim:self.end_dim]
        prev_action = env.action_manager.prev_action[:, self.start_dim:self.end_dim]
        dt2 = (env.step_dt)**2
        error = torch.square(action - prev_action) / dt2
        return torch.sum(error, dim=1)


class action_smoothness2(ManagerTermBase):

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

        action = env.action_manager.action[:, self.start_dim:self.end_dim]
        prev_action = env.action_manager.prev_action[:, self.start_dim:self.end_dim]
        prev_prev_action = self.prev_prev_action
        dt2 = (env.step_dt)**2
        error = torch.square(action - 2*prev_action + prev_prev_action)/dt2
        return -torch.sum(error, dim=1)

def joint_regularization(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = torch.exp(-torch.square(error) / 0.25)
    return torch.mean(error, dim=1)

def orientation_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking upright orientation
    error = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    error = torch.exp(-torch.square(error / 0.2) / 0.25)
    return torch.mean(error, dim=1)

def track_lin_vel_reward(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    commands = env.command_manager.get_command(command_name)[:, :2]
    error = commands - vel_yaw[:, :2]
    error *= 1./(1. + torch.abs(commands))
    return torch.exp(-torch.square(error) / std**2).sum(dim=-1)

def track_ang_vel_reward(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (z axis) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)[:, :2]
    error = commands[:, 2] - asset.data.root_ang_vel_b[:, 2]
    error *= 1./(1. + torch.abs(commands))
    return torch.exp(-torch.square(error) / std**2)

def joint_position_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 0.5) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    return torch.sum(torch.square(error), dim=1) * \
        torch.exp(-torch.square(torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1)) / std**2)

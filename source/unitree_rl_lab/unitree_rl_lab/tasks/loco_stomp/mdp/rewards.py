from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

from unitree_rl_lab.tasks.loco_stomp.mdp import commands

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import track_lin_vel_xy_yaw_frame_exp, feet_slide

def reward_zero_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    ang_vel_z_error = torch.abs(asset.data.root_ang_vel_b[:, 2] / std)
    reward = torch.exp(-ang_vel_z_error)

    return reward - ang_vel_z_error * 0.25


def action_rate_l2_ext(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, asset_cfg.joint_ids] - env.action_manager.prev_action[:, asset_cfg.joint_ids]), dim=1)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

def stand_deviation_l1(env: ManagerBasedRLEnv,
                       command_name: str = "base_velocity",
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    pos_error = (asset.data.joint_pos - asset.data.default_joint_pos)[:, asset_cfg.joint_ids]

    cmd: commands.StompCommand = env.command_manager.get_term(command_name)

    is_walking = torch.logical_not(cmd.is_standing_env)

    pos_error[is_walking, :] = 0.0
    return torch.sum(torch.abs(pos_error), dim=1)

def feet_gait(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name="stomp_command",
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    cmd: commands.StompCommand = env.command_manager.get_term(command_name)

    is_stance = cmd.feet_global_phases < cmd.cfg.threshold

    reward = ~(is_stance ^ is_contact)
    return torch.sum(reward, dim=-1)

def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,

    std: float,
    max_height: float,
    target_height: float,
    speed: float,

    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""

    cmd: commands.StompCommand = env.command_manager.get_term(command_name)

    swing_phase = cmd.feet_swing_phases

    target_height = torch.sin(swing_phase * torch.pi) * target_height
    target_height = torch.clamp_min(target_height, min=0.0)

    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # feet_error = torch.clamp_max(feet_z - target_height, max=0) # allow feet to be higher than target, but penalize if they are lower
    feet_error = feet_z - target_height
    feet_error[target_height > 0.008] = torch.clamp_max(feet_error[target_height > 0.008], max=0) # only penalize when target height is above 0 (i.e. during swing phase)

    feet_exp = torch.exp(- torch.abs(feet_error) / std)
    return torch.mean(feet_exp, dim=-1)


def com_zero(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        std: float
    ) -> torch.Tensor:

    std = max(std, 0.1)

    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids]

    quat_w = torch.repeat_interleave(asset.data.root_link_quat_w[:, None, :], pos.shape[1], dim=1)

    try:
        pos_b = math_utils.quat_apply_inverse(quat_w, pos)
        # pos = torch.mean(pos_b[:, :, :2], dim=1)
        com_b = math_utils.quat_apply_inverse(asset.data.root_link_quat_w, asset.data.root_com_pos_w)
    except:
        pos_b = math_utils.quat_rotate_inverse(quat_w, pos)
        # pos = torch.mean(pos_b[:, :, :2], dim=1)
        com_b = math_utils.quat_rotate_inverse(asset.data.root_link_quat_w, asset.data.root_com_pos_w)

    left_ankle = pos_b[:, 0, :2]
    right_ankle = pos_b[:, 1, :2]

    support_vec = right_ankle - left_ankle
    support_len = torch.linalg.norm(support_vec, dim=1)
    support_len = torch.clamp(support_len, min=0.05)

    support_dir = support_vec / (support_len.unsqueeze(1) + 1e-6)
    normal_vec = torch.stack([-support_dir[:, 1], support_dir[:, 0]], dim=1)

    com_proj = com_b[:, :2] - left_ankle
    com_dist = torch.sum(com_proj * normal_vec, dim=1)
    com_dist = torch.clamp(com_dist, min=-0.5, max=0.5)

    com_along = torch.sum(com_proj * support_dir, dim=1)

    margin = support_len * 0.1
    # stable = (com_along >= -margin) & (com_along <= (support_len + margin))

    dist_reward = torch.exp(- torch.square((com_dist * 2 - std) / std))
    # dist_reward = torch.exp(- torch.abs(com_dist) / std)
    # range_reward = torch.where(stable, 1.0, 0.2)
    range_reward = torch.sigmoid(
                (com_along + margin) / (0.1 + support_len)
            ) * torch.sigmoid(
                (support_len + margin - com_along) / (0.1 + support_len)
            )

    total_reward = dist_reward * range_reward
    return total_reward


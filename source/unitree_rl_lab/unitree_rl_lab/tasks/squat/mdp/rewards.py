
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

from unitree_rl_lab.tasks.squat.mdp import command_squat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def track_squat_pos_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    command: command_squat.SquatCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    pos_error = command.command_pos - asset.data.joint_pos[:, asset_cfg.joint_ids]
    pos_error = torch.abs(pos_error / std)
    reward = torch.sum(torch.exp(-pos_error), dim = -1)
    return  reward

def track_squat_error(
    env: ManagerBasedRLEnv, std: float, command_name: str,
    finished_weight: float = 1, finished_max_weight: float = 2,
    penalty_weight: float = 1, penalty_max_weight: float = 3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    command: command_squat.SquatCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    pos_error = command.command_pos - asset.data.joint_pos[:, asset_cfg.joint_ids]

    pos_error = torch.abs(pos_error / std)
    error = torch.zeros_like(pos_error)
    error[command.is_finished_flags] = pos_error[command.is_finished_flags] * finished_weight
    pos_error[command.is_finished_flags] *= finished_weight
    return  torch.sum(torch.square(pos_error), dim = -1) * penalty_weight


def track_symmetry_pos_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    pos_error = pos[:, 0::2] - pos[:, 1::2]
    #pos_error = torch.square(pos_error / std)
    #return torch.norm(torch.exp(-pos_error), dim = -1)
    pos_error = torch.abs(pos_error / std)
    return torch.sum(torch.exp(-pos_error), dim = -1)


def track_constraint_width(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_width: float = 0.2,
    std: float = 0.04
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]

    quat_w = torch.repeat_interleave(asset.data.root_link_quat_w[:, None, :], body_pos_w.shape[1], dim=1)

    body_pos = math_utils.quat_apply_inverse(quat_w, body_pos_w)

    #error = torch.square((torch.abs(body_pos[:, 0::2, 1] - body_pos[:, 1::2, 1]) - target_width) / std)
    error = torch.abs((torch.abs(body_pos[:, 0::2, 1] - body_pos[:, 1::2, 1]) - target_width) / std)

    # 将偏差放大（乘以 100），计算其指数惩罚，最后求所有对的平均值作为最终 reward
    return - torch.norm(error, dim=-1) + torch.norm(torch.exp(- error), dim = -1)

def reward_pitch2zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float = 0.25
) -> torch.Tensor:
    # 相位计算（与 reward_orientation 相同）
    command: command_squat.SquatCommand = env.command_manager.get_term(command_name)
    phase = (command.squat_phase + (0.5 * torch.pi  - command.init_phase)) / torch.pi
    phase = torch.clamp_max(phase, max = 0.5)
    pitch_compensation = torch.abs(0.5 -  phase[:, 0]) * std

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # 左右腿关节位置求和
    left = torch.square((torch.sum(joint_pos[:, 0::2], dim = -1) + pitch_compensation) / (std * 0.4))
    right = torch.square((torch.sum(joint_pos[:, 1::2], dim = -1) + pitch_compensation) / (std * 0.4))

    return torch.exp(- left) + torch.exp(- right)

def reward_pitch_forward_sing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    left = torch.sign(torch.sum(joint_pos[:, 0::2], dim = -1))
    right = torch.square(torch.sum(joint_pos[:, 1::2], dim = -1))

    return left + right

def com_zero(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        std: float
    ) -> torch.Tensor:

    command: command_squat.SquatCommand = env.command_manager.get_term(command_name)

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

    # only for static
    total_reward[command.is_finished_flags == 0] = 0
    return total_reward

def contact_same_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids])

    forces = torch.sum(torch.abs((forces[:, 0] - forces[:, 1]) / 20), dim=1)
    # forces = torch.sum(torch.square((forces[:, 0] - forces[:, 1]) / 20), dim=1)
    return torch.exp(- forces )

def reward_zero_ang_vel_exp(
    env: ManagerBasedRLEnv, std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # ang_vel_error = torch.sum(torch.square(asset.data.root_ang_vel_b / std), dim = 1)
    ang_vel_error = torch.norm(asset.data.root_ang_vel_b / std, dim = 1)
    reward = torch.exp(-ang_vel_error)
    return reward

def reward_zero_ang_vel_exp_v1(
    env: ManagerBasedRLEnv, std: float, command_name: str = "squat_command",
    finished_weight: float = 20, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    command: command_squat.SquatCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    # ang_vel_error = torch.sum(torch.square(asset.data.root_ang_vel_b / std), dim = 1)
    ang_vel_error = torch.norm(asset.data.root_ang_vel_b / std, dim = 1)
    reward = torch.exp(-ang_vel_error)
    ang_vel_error[command.is_finished_flags] *= finished_weight
    return reward - ang_vel_error * 0.25

def reward_zero_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    # lin_vel_error = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2] / std), dim = 1)
    lin_xy_error = torch.norm(asset.data.root_lin_vel_b[:, :2] / std, dim = 1)
    reward = torch.exp(-lin_xy_error)
    return reward

def penalty_lin_vel_z_v1(
    env: ManagerBasedRLEnv, command_name: str = "squat_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    command: command_squat.SquatCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    lin_z_error = torch.abs(asset.data.root_lin_vel_w[:, 2])
    lin_z_error[command.is_finished_flags == 0] = 0
    return lin_z_error

def reward_zero_lin_vel_xy_exp_v1(
    env: ManagerBasedRLEnv, std: float, command_name: str = "squat_command",
    finished_weight: float = 20, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    command: command_squat.SquatCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    # lin_vel_error = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2] / std), dim = 1)
    lin_xy_error = torch.norm(asset.data.root_lin_vel_b[:, :2] / std, dim = 1)
    lin_z_error = torch.abs(asset.data.root_lin_vel_b[:, 2])
    reward = torch.exp(-lin_xy_error)
    lin_xy_error[command.is_finished_flags] *= finished_weight
    lin_xy_error[command.is_finished_flags] += lin_z_error[command.is_finished_flags] * finished_weight
    return reward - lin_xy_error * 0.25


def joint_vel_l2(env: ManagerBasedRLEnv, command_name: str = "squat_command", finished_weight: float = 20,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    command: command_squat.SquatCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    penalty[command.is_finished_flags] *=  finished_weight
    return penalty


def joint_acc_l2(env: ManagerBasedRLEnv, command_name: str = "squat_command", finished_weight: float = 20,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    command: command_squat.SquatCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
    penalty[command.is_finished_flags] *=  finished_weight
    return penalty


def energy(env: ManagerBasedRLEnv, command_name: str = "squat_command", finished_weight: float = 20,
           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    command: command_squat.SquatCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)
    penalty[command.is_finished_flags] *=  finished_weight
    return penalty

def reward_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float = 0.25
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    '''
    由 pi / 2 转为  command.init_phase 为站立；
    补偿解为 0

    到 - pi / 2 最大，为深蹲。 补偿最大

    站立姿势 0.5 ~ init_phase

    target_compensation
    sin(0) = 0
    期望机器人完全直立 pitch_b = 0

    深蹲姿势 phase = - pi / 2

    target_compensation = abs(- 0.5 - command.init_phase / (0.5 * pi)) *  std
    期望机器人轻微前倾以保持平衡
    '''
    # 相位计算
    command: command_squat.SquatCommand = env.command_manager.get_term(command_name)
    phase = (command.squat_phase + (0.5 * torch.pi  - command.init_phase)) / torch.pi
    phase = torch.clamp_max(phase, max = 0.5)
    # 补偿位姿
    target_compensation = torch.abs(0.5 -  phase[:, 0]) * std

    pitch_b = asset.data.projected_gravity_b[:, 0] # x project
    error = torch.abs(torch.sin(target_compensation) - pitch_b)
    return torch.exp(- error / 0.1)

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

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import track_lin_vel_xy_yaw_frame_exp


def reward_zero_lin_vel_xy_exp(
    env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_error = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1) / std
    reward = torch.exp(-lin_vel_error)

    return reward - lin_vel_error * 0.25

def reward_zero_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    ang_vel_z_error = torch.abs(asset.data.root_ang_vel_b[:, 2] / std)
    reward = torch.exp(-ang_vel_z_error)

    return reward - ang_vel_z_error * 0.25

def reward_track_pitch(
    env: ManagerBasedRLEnv,
    command_name: str,

    std: float,

    max_stomp: float,
    target_stomp: float,
    speed: float,

    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""

    cmd: commands.StompCommand = env.command_manager.get_term(command_name)

    swing_phase = cmd.feet_swing_phases

    swing_stomp_target = torch.sin(swing_phase * torch.pi) * target_stomp

    asset: Articulation = env.scene[asset_cfg.name]
    stomp = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    error = stomp - swing_stomp_target
    diff = torch.square(error / std)

    stomp_exp = torch.exp(- diff)
    return torch.mean(stomp_exp, dim=-1)

def penalize_track_pitch(
    env: ManagerBasedRLEnv,
    command_name: str,

    std: float,

    max_stomp: float,
    target_stomp: float,
    speed: float,

    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""

    cmd: commands.StompCommand = env.command_manager.get_term(command_name)

    swing_phase = cmd.feet_swing_phases

    swing_stomp_target = torch.sin(swing_phase * torch.pi) * target_stomp

    asset: Articulation = env.scene[asset_cfg.name]
    stomp = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    error = stomp - swing_stomp_target
    diff = torch.square(error / std)
    return torch.mean(diff, dim=-1)


def reward_foot_clearance(
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

    swing_target0 = torch.sin(swing_phase * torch.pi) * target_height
    swing_target = torch.clamp_min(swing_target0, min=0.0)

    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - 0.006  ##
    # feet_error = torch.clamp_max(feet_z - target_height, max=0) # allow feet to be higher than target, but penalize if they are lower
    feet_error = feet_z - swing_target

    clamp_mask = swing_target > 0.008
    feet_error[clamp_mask] = torch.clamp_max(feet_error[clamp_mask], max=0) # only penalize when target height is above 0 (i.e. during swing phase)

    diff = torch.square(feet_error / std)

    feet_exp = torch.exp(- diff)
    return torch.mean(feet_exp, dim=-1)


def penalize_foot_clearance(
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

    swing_target0 = torch.sin(swing_phase * torch.pi) * target_height
    swing_target = torch.clamp_min(swing_target0, min=0.0)

    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - 0.006  ##
    # feet_error = torch.clamp_max(feet_z - target_height, max=0) # allow feet to be higher than target, but penalize if they are lower
    feet_error = feet_z - swing_target

    clamp_mask = swing_target > 0.008
    feet_error[clamp_mask] = torch.clamp_max(feet_error[clamp_mask], max=0) # only penalize when target height is above 0 (i.e. during swing phase)

    diff = torch.square(feet_error / std)
    return torch.mean(diff, dim=-1)


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

    return - torch.norm(error, dim=-1) + torch.norm(torch.exp(- error), dim = -1)


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

def track_feet_gait(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name="stomp_command",
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.005

    cmd: commands.StompCommand = env.command_manager.get_term(command_name)

    is_stance = cmd.feet_global_phases < cmd.cfg.threshold

    reward = ~(is_stance ^ is_contact)
    return torch.sum(reward, dim=-1)

def penalize_feet_gait(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name="stomp_command",
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.005

    cmd: commands.StompCommand = env.command_manager.get_term(command_name)

    is_swing = cmd.feet_global_phases > cmd.cfg.threshold

    penalize = (is_swing ^ is_contact)
    return torch.sum(penalize, dim=-1)

def com_support(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        std: float,
        target_width: float = 0.2,
        command_name: str ="stomp_command",
        min_reward: float = 0.05,  #
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

    ankle = pos_b[:, :, :2]                                 # N * 2 * 2
    dis = torch.norm(ankle - com_b[:, None, :2], dim=-1)    # N * 2

    cmd: commands.StompCommand = env.command_manager.get_term(command_name)
    swing_dis = (1 - torch.sin(cmd.feet_swing_phases * torch.pi)) * target_width *  0.5

    diff = torch.abs(dis - swing_dis) / std

    swing_status = cmd.feet_global_phases > cmd.cfg.threshold     # N * 2

    # 判定「单腿摆动、另一条腿支撑」：异或结果为True（形状[N]）
    # 双足机器人：swing_status=[True, False]或[False, True] → with_swing=True
    with_swing = swing_status[:, 0] ^ swing_status[:, 1]             # N
    # 判定支撑相（摆动相取反，形状[N, 2]）
    support_phases = ~swing_status     # N * 2
    # 非单腿支撑时（站立/双腿支撑/双腿摆动），强制支撑相为False
    support_phases[~with_swing] = False

    # 仅保留支撑腿的质心-脚踝距离，摆动腿的距离置0
    diff[~support_phases] = 0.0
    # 求和：每个环境实例的支撑腿距离（形状[N]）
    diff = torch.sum(diff, dim=-1)

    # 指数奖励：距离越小，奖励越接近1
    reward = torch.exp(- diff / std)
    # 非单腿支撑时，奖励强制为0
    reward[~with_swing] = 0
    return reward


def feet_slide(env,
    command_name: str = "stomp_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    cmd: commands.StompCommand = env.command_manager.get_term(command_name)
    stand_phases = cmd.feet_global_phases < cmd.cfg.threshold     # N * 2

    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    reward = torch.sum(body_vel.norm(dim=-1) * stand_phases.float(), dim= -1)
    return reward


def reward_euler(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """
    Calculates the reward for maintaining a flat base orientation. It penalizes deviation
    from the desired base orientation using the base euler angles and the projected gravity vector.

    Args:
        env (ManagerBasedRLEnv): The environment instance, which contains the simulation scene
            and provides access to the assets and their states.
        asset_cfg (SceneEntityCfg): Configuration for the asset whose orientation is being evaluated.
            Defaults to a configuration with the name "robot".

    Returns:
        torch.Tensor: The calculated reward value, which is a combination of penalties for
        deviations in Euler angles and the projected gravity vector.
    """
    # Extract the asset from the environment using the provided configuration
    asset: Articulation = env.scene[asset_cfg.name]

    # Get the root quaternion of the asset
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # Assuming the root link is the first body in the asset

    # Convert the quaternion to Euler angles (roll, pitch, yaw)
    roll0, pitch0, yaw = math_utils.euler_xyz_from_quat(quat[:, 0, :], wrap_to_2pi=True)  # Assuming the root link is the first body in the asset
    roll1, pitch1, yaw = math_utils.euler_xyz_from_quat(quat[:, 1, :], wrap_to_2pi=True)  # Assuming the root link is the first body in the asset

    euler_mismatch0 = torch.exp(-(torch.abs(roll0) + torch.abs(pitch0)) * 10)
    euler_mismatch1 = torch.exp(-(torch.abs(roll1) + torch.abs(pitch1)) * 10)

    # Combine the two mismatch values into a single reward (average of both components)
    return (euler_mismatch0 + euler_mismatch1) / 2.


def penalty_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    gw = torch.repeat_interleave(asset.data.GRAVITY_VEC_W.unsqueeze(1), repeats=len(asset_cfg.body_ids), dim=1)
    body_gravity_b = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids], gw)
    return torch.sum(torch.sum(torch.square(body_gravity_b[:, :, :2]), dim=-1), dim=-1)

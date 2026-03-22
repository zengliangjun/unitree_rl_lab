from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from unitree_rl_lab.tasks.locomotionv2.mdp import commands
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def action_rate_l2_ext(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, asset_cfg.joint_ids] - env.action_manager.prev_action[:, asset_cfg.joint_ids]), dim=1)

def stand_deviation_l1(env: ManagerBasedRLEnv,
                       command_name: str = "base_velocity",
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    """Penalize joint positions that deviate from the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos_error = (asset.data.joint_pos - asset.data.default_joint_pos)[:, asset_cfg.joint_ids]

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

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

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

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

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    is_swing = cmd.feet_global_phases > cmd.cfg.threshold

    penalize = (is_swing ^ is_contact)
    return torch.sum(penalize, dim=-1)

def penalty_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    gw = torch.repeat_interleave(asset.data.GRAVITY_VEC_W.unsqueeze(1), repeats=len(asset_cfg.body_ids), dim=1)
    body_gravity_b = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids], gw)
    return torch.sum(torch.sum(torch.square(body_gravity_b[:, :, :2]), dim=-1), dim=-1)


def feet_slide(env,
    command_name: str = "stomp_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)
    stand_phases = cmd.feet_global_phases < cmd.cfg.threshold     # N * 2

    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    reward = torch.sum(body_vel.norm(dim=-1) * stand_phases.float(), dim= -1)
    return reward


def reward_feet_width(
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

def penalize_feet_forces(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg,
   threshold: float = 500,
   max_over_penalize_forces: float = 400) -> torch.Tensor:
    """
    Penalizes excessive contact forces on the feet to discourage high impact.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Sensor configuration including sensor name and body IDs.
        threshold (float): Force threshold above which penalties are applied.
        max_over_penalize_forces (float): Maximum force value to clip the penalty.

    Returns:
        torch.Tensor: The calculated penalty as the sum of clamped excessive forces from specified body parts.
    """
    # Retrieve the contact sensor instance.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Compute the norm of forces for specified body parts.
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    # Calculate penalty by clamping forces exceeding the threshold.
    _reward = torch.clamp(forces - threshold, min=0, max=max_over_penalize_forces)
    # Sum penalties across all relevant body parts.
    _reward = torch.sum(_reward, dim=-1)
    return _reward

def penalize_feet_forces_v2(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg,
   threshold: float = 500,
   contact_time_threshold: float = 0.07,
   max_over_penalize_forces: float = 400) -> torch.Tensor:
    """
    Penalizes excessive contact forces on the feet to discourage high impact.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Sensor configuration including sensor name and body IDs.
        threshold (float): Force threshold above which penalties are applied.
        max_over_penalize_forces (float): Maximum force value to clip the penalty.

    Returns:
        torch.Tensor: The calculated penalty as the sum of clamped excessive forces from specified body parts.
    """
    # Retrieve the contact sensor instance.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # contact_flags
    contact_flags = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] < contact_time_threshold

    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    # Calculate penalty by clamping forces exceeding the threshold.
    _reward = torch.clamp(forces - threshold, min=0, max=max_over_penalize_forces) * contact_flags.float()
    # Sum penalties across all relevant body parts.
    _reward = torch.sum(_reward, dim=-1)
    return _reward

def penalize_feet_forces_v2(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg,
   threshold: float = 500,
   contact_time_threshold: float = 0.07,
   max_over_penalize_forces: float = 400) -> torch.Tensor:
    """
    Penalizes excessive contact forces on the feet to discourage high impact.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Sensor configuration including sensor name and body IDs.
        threshold (float): Force threshold above which penalties are applied.
        max_over_penalize_forces (float): Maximum force value to clip the penalty.

    Returns:
        torch.Tensor: The calculated penalty as the sum of clamped excessive forces from specified body parts.
    """
    # Retrieve the contact sensor instance.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # contact_flags
    contact_flags = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] < contact_time_threshold

    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    # Calculate penalty by clamping forces exceeding the threshold.
    _reward = torch.clamp(forces - threshold, min=0, max=max_over_penalize_forces) * contact_flags.float()
    # Sum penalties across all relevant body parts.
    _reward = torch.sum(_reward, dim=-1)
    return _reward



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

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    swing_phase = cmd.feet_swing_phases

    swing_target0 = torch.sin(swing_phase * torch.pi) * target_height
    swing_target = torch.clamp_min(swing_target0, min=0.0)

    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  ##
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

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    swing_phase = cmd.feet_swing_phases

    swing_target0 = torch.sin(swing_phase * torch.pi) * target_height
    swing_target = torch.clamp_min(swing_target0, min=0.0)

    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  ##
    # feet_error = torch.clamp_max(feet_z - target_height, max=0) # allow feet to be higher than target, but penalize if they are lower
    feet_error = feet_z - swing_target

    clamp_mask = swing_target > 0.008
    feet_error[clamp_mask] = torch.clamp_max(feet_error[clamp_mask], max=0) # only penalize when target height is above 0 (i.e. during swing phase)

    diff = torch.square(feet_error / std)
    return torch.mean(diff, dim=-1)

def reward_foot_clearance_v2(
    env: ManagerBasedRLEnv,
    command_name: str,

    std: float,
    max_height: float,
    target_height: float,
    speed: float,

    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    swing_phase = torch.logical_and(cmd.feet_swing_phases > 0.008, cmd.feet_swing_phases < 0.992)

    #swing_target0 = torch.sin(swing_phase * torch.pi) * target_height
    #swing_target = torch.clamp_min(swing_target0, min=0.0)
    swing_target = swing_phase.float() * target_height

    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  ##
    # feet_error = torch.clamp_max(feet_z - target_height, max=0) # allow feet to be higher than target, but penalize if they are lower
    feet_error = feet_z - swing_target

    clamp_mask = swing_target > 0.008
    feet_error[clamp_mask] = torch.clamp_max(feet_error[clamp_mask], max=0) # only penalize when target height is above 0 (i.e. during swing phase)

    diff = torch.square(feet_error / std)

    feet_exp = torch.exp(- diff)
    return torch.mean(feet_exp, dim=-1)


def penalize_foot_clearance_v2(
    env: ManagerBasedRLEnv,
    command_name: str,

    std: float,
    max_height: float,
    target_height: float,
    speed: float,

    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    swing_phase = torch.logical_and(cmd.feet_swing_phases > 0.008, cmd.feet_swing_phases < 0.992)

    # swing_target0 = torch.sin(swing_phase * torch.pi) * target_height
    # swing_target = torch.clamp_min(swing_target0, min=0.0)
    swing_target = swing_phase.float() * target_height

    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  ##
    # feet_error = torch.clamp_max(feet_z - target_height, max=0) # allow feet to be higher than target, but penalize if they are lower
    feet_error = feet_z - swing_target

    clamp_mask = swing_target > 0.008
    feet_error[clamp_mask] = torch.clamp_max(feet_error[clamp_mask], max=0) # only penalize when target height is above 0 (i.e. during swing phase)

    diff = torch.square(feet_error / std)
    return torch.mean(diff, dim=-1)


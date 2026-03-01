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
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name="stomp_command",
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    cmd: commands.StompCommand = env.command_manager.get_term(command_name)

    is_stance = leg_phase < threshold
    is_stance[cmd.is_standing_env] = True  # if standing, all legs should be in stance

    reward = ~(is_stance ^ is_contact)
    return torch.sum(reward, dim=-1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    threshold: float,
    command_name: str,
    asset_cfg: SceneEntityCfg, target_height: float, std: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    cmd: commands.StompCommand = env.command_manager.get_term(command_name)

    is_stance = leg_phase < threshold
    is_stance[cmd.is_standing_env] = True  # if standing, all legs should be in stance

    target_height = torch.sin((leg_phase + 0.5) * 2 * torch.pi) * target_height
    target_height = torch.clamp_min(target_height, min=0.0)
    target_height[cmd.is_standing_env] = 0

    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.norm(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height, dim=-1)
    return torch.exp(-error / std)



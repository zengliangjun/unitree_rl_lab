from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import numpy as np

def shoulder_gait_penalty(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    shoulder_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    swing_range: float = 0.3,
    command_name: str = "base_velocity"
) -> torch.Tensor:

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    phase = torch.cat(phases, dim=-1) * np.pi

    #
    swing_target = torch.abs(torch.cos(phase) * swing_range)

    #
    asset: Articulation = env.scene[shoulder_cfg.name]
    hip_sign = torch.sign(asset.data.joint_pos[:, hip_cfg.joint_ids] - asset.data.default_joint_pos[:, hip_cfg.joint_ids]) * -1
    swing_target *= hip_sign

    #
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1, keepdim=True)
    swing_target *= cmd_norm > 0.1

    shoulder_pos = asset.data.joint_pos[:, shoulder_cfg.joint_ids] - asset.data.default_joint_pos[:, shoulder_cfg.joint_ids]

    penalty = torch.sum(torch.square(shoulder_pos - swing_target), dim = -1)

    return penalty

'''
S -> leg is standing (stance)
F -> leg is swinging (swing)

SSSSSSSSSFFFFFFFFF

B -> shoulder is back
F -> shoulder is front

BBBBBFFFFFFFFFFBBBBB

'''
def penalty_shoulder_gait_signwithlinevel(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    asset_cfg: SceneEntityCfg,
    swing_range: float = 0.25,
    cent_pos: float = 0.15,
    command_name: str = "base_velocity"
) -> torch.Tensor:

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    phase = torch.cat(phases, dim=-1) * np.pi * 2

    cmd = env.command_manager.get_command(command_name)

    swing_sign = torch.sign(cmd[:, :1])

    cmd_norm = torch.norm(cmd, dim=1, keepdim=True)
    scale = torch.abs(cmd[:, :1]) / (cmd_norm + 1e-6)

    ###
    swing_target = torch.cos(phase) * scale * swing_range * swing_sign + cent_pos  # n * 2

    is_stand = cmd_norm[:, 0] < 0.1
    #
    asset: Articulation = env.scene[asset_cfg.name]

    swing_target[is_stand, :] = asset.data.default_joint_pos[:, asset_cfg.joint_ids][is_stand, :]
    #
    pos_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - swing_target
    penalty_error = torch.sum(torch.square(pos_error), dim = -1)
    return penalty_error

def penalty_knee(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.55,
    command_name: str = "base_velocity"
) -> torch.Tensor:

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    phase = torch.cat(phases, dim=-1)

    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    is_stand = cmd_norm < 0.1

    # stand_phase = phase < threshold
    swing_phase = phase > threshold
    #
    asset: Articulation = env.scene[asset_cfg.name]
    pos_error = (asset.data.joint_pos[:, asset_cfg.joint_ids]).clone()

    pos_error[swing_phase] = (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids] * 1.5)[swing_phase] * 2
    pos_error[is_stand] = (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids])[is_stand]
    #

    penalty_error = torch.sum(torch.square(pos_error), dim = -1)
    return penalty_error

def stand_deviation_l1(env: ManagerBasedRLEnv,
                       command_name: str = "base_velocity",
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    pos_error = (asset.data.joint_pos - asset.data.default_joint_pos)[:, asset_cfg.joint_ids]

    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    is_walking = cmd_norm > 0.1
    pos_error[is_walking, :] = 0.0
    return torch.sum(torch.abs(pos_error), dim=1)

def joint_deviation_l4(env: ManagerBasedRLEnv,
                       command_name: str = "base_velocity",
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                       std: float = 0.25) -> torch.Tensor:

    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    pos_error = (asset.data.joint_pos - asset.data.default_joint_pos)[:, asset_cfg.joint_ids] / std

    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    is_stand = cmd_norm < 0.1
    pos_error[is_stand, :] = 0.0

    return torch.sum(torch.pow(pos_error, 4), dim=1)


##############################################################################################
def reward_mismatch_vel_exp(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    linear_weight = 10,
    angle_weight = 5):
    """
    Computes reward based on mismatches in linear and angular velocities.
    Uses exponential kernels to reward deviations from stability.

    Args:
        env (ManagerBasedRLEnv): Environment instance.
        asset_cfg (SceneEntityCfg): Asset configuration.
        linear_weight: Weight factor for linear velocity mismatch.
        angle_weight: Weight factor for angular velocity mismatch.

    Returns:
        torch.Tensor: Combined reward of velocity mismatches.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Exponential penalty on vertical linear velocity error
    lin_mismatch = torch.exp(-torch.square(asset.data.root_lin_vel_b[:, 2]) * linear_weight)
    # Exponential penalty on angular velocity error in x-y plane
    ang_mismatch = torch.exp(-torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1) * angle_weight)
    # Combine both penalties
    return (lin_mismatch + ang_mismatch) / 2.


def reward_mismatch_speed(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = 'base_velocity',
    ):
    """
    Rewards or penalizes based on the robot's speed relative to the commanded speed.

    Args:
        env (ManagerBasedRLEnv): Environment instance.
        asset_cfg (SceneEntityCfg): Asset configuration.
        command_name (str): Name of the command containing desired base velocity.

    Returns:
        torch.Tensor: Speed reward where deviations and direction mismatches are penalized.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Compute absolute base linear speed in x-direction
    absolute_speed = torch.abs(asset.data.root_lin_vel_b[:, 0])
    # Get commanded velocity and compute its absolute value
    command = env.command_manager.get_command(command_name)
    absolute_command = torch.abs(command[:, 0])

    # Determine if current speed is too low, too high, or within desired range
    speed_too_low = absolute_speed < 0.5 * absolute_command
    speed_too_high = absolute_speed > 1.2 * absolute_command
    speed_desired = ~(speed_too_low | speed_too_high)

    # Detect if movement direction mismatches the command
    sign_mismatch = torch.sign(asset.data.root_lin_vel_b[:, 0]) != torch.sign(command[:, 0])

    reward = torch.zeros_like(absolute_speed)
    # Penalize low speed
    reward[speed_too_low] = -1.0
    # Neutral reward for too high speed
    reward[speed_too_high] = 0.
    # Positive reward for desired speed range
    reward[speed_desired] = 1.2
    # Highest penalty if direction is incorrect
    reward[sign_mismatch] = -2.0
    # Only apply reward if command is significant
    return reward * (torch.abs(command[:, 0]) > 0.1)


def reward_track_vel_hard(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = 'base_velocity'):
    """
    Calculates reward for accurately tracking linear (xy) and angular (yaw) velocity commands.

    Args:
        env (ManagerBasedRLEnv): Environment instance.
        asset_cfg (SceneEntityCfg): Asset configuration.
        command_name (str): Command name for base velocity.

    Returns:
        torch.Tensor: Reward computed from tracking errors in velocity commands.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Compute error for linear velocity on xy axes
    lin_vel_error = torch.norm(command[:, :2] - asset.data.root_lin_vel_b[:, :2], dim=1)
    lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

    # Compute error for angular velocity (yaw)
    ang_vel_error = torch.abs(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

    # Apply extra penalty proportional to total linear error
    linear_error = 0.2 * (lin_vel_error + ang_vel_error)

    return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error


class reward_base_acc(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # Parameter docstrings and inline comments sufficiently explain the logic.
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.prev_root_lin_vel_b = torch.zeros_like(asset.data.root_lin_vel_b)
        self.prev_root_ang_vel_b = torch.zeros_like(asset.data.root_ang_vel_b)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Reset stored velocities for specified environments
        if len(env_ids) == 0:
            return
        self.prev_root_lin_vel_b[env_ids] = 0
        self.prev_root_ang_vel_b[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """
        Computes a reward based on penalizing high base accelerations.
        Encourages smoother motion by comparing current and previous velocities.

        Args:
            env (ManagerBasedRLEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Asset configuration, default is "robot".

        Returns:
            torch.Tensor: Reward value computed from the base's acceleration.
        """
        asset: Articulation = env.scene[asset_cfg.name]

        # Compute difference between previous and current velocities (acceleration)
        root_acc = self.prev_root_lin_vel_b - asset.data.root_lin_vel_b
        ang_acc = self.prev_root_ang_vel_b - asset.data.root_ang_vel_b

        # Exponential penalty based on norm of acceleration (both linear and angular)
        rew = torch.exp(-(torch.norm(root_acc, dim=1) * 2 + torch.norm(ang_acc, dim=1)))

        # Update stored velocities for the next call
        self.prev_root_lin_vel_b[...] = asset.data.root_lin_vel_b
        self.prev_root_ang_vel_b[...] = asset.data.root_ang_vel_b

        return rew

def feet_clearance(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = 'base_velocity',
    period: float = 0.8,
    offset: list[float] = [0.0, 0.5],
    stand_threshold: float = 0.55,
    swing_height: float = 0.08,
    tracking_sigma: float = 0.008) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    phase = torch.cat(phases, dim=-1)


    def _expected_foot_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
        """Expected foot height from gait phase using a cubic Bézier profile."""

        def cubic_bezier_interpolation(y_start: torch.Tensor, y_end: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            y_diff = y_end - y_start
            bezier = x**3 + 3 * (x**2 * (1 - x))
            return y_start + y_diff * bezier

        x = phi
        stance = cubic_bezier_interpolation(torch.zeros_like(x), torch.full_like(x, swing_height), 2 * x)
        swing = cubic_bezier_interpolation(torch.full_like(x, swing_height), torch.zeros_like(x), 2 * x - 1)
        return torch.where(x <= stand_threshold, stance, swing)

    rz_left = _expected_foot_height(phase[:, 0], swing_height)
    rz_right = _expected_foot_height(phase[:, 1], swing_height)

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    # Calculate height tracking errors
    error_left = torch.square(foot_z[:, 0] - rz_left)
    error_right = torch.square(foot_z[:, 1] - rz_right)
    # Combine errors and apply exponential reward
    total_error = error_left + error_right
    return torch.exp(-total_error / tracking_sigma)

def penalty_feet_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    valid_len = env.episode_length_buf > 50

    asset: Articulation = env.scene[asset_cfg.name]
    gw = torch.repeat_interleave(asset.data.GRAVITY_VEC_W.unsqueeze(1), repeats=len(asset_cfg.body_ids), dim=1)
    feet_gravity_b = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids], gw)
    return torch.sum(torch.sum(torch.square(feet_gravity_b[:, :, :2]), dim=-1), dim=-1) * valid_len.float()

def action_rate_l2_ext(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, asset_cfg.joint_ids] - env.action_manager.prev_action[:, asset_cfg.joint_ids]), dim=1)



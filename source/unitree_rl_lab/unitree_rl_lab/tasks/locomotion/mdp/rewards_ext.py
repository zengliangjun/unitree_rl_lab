from __future__ import annotations

import torch
from typing import TYPE_CHECKING

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
    threshold: float = 0.5,
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

    pos_error[swing_phase] = (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids] * 1.5)[swing_phase]
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

class action_rate_l2_withname(ManagerTermBase):

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

        error = torch.square(env.action_manager.action - env.action_manager.prev_action)[:, self.start_dim: self.end_dim]
        return torch.sum(error, dim=1)

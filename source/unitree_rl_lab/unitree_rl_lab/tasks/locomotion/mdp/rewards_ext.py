from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

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
    phase = torch.cat(phases, dim=-1) * np.pi   # n * 2

    swing_sign = torch.sign(env.command_manager.get_command(command_name)[:, :1])
    swing_target = torch.cos(phase) * swing_range * swing_sign + cent_pos  # n * 2

    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_stand = cmd_norm > 0.1
    #
    asset: Articulation = env.scene[asset_cfg.name]

    swing_target[is_stand, :] = asset.data.default_joint_pos[:, asset_cfg.joint_ids][is_stand, :]
    #
    pos_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - swing_target
    penalty_error = torch.sum(torch.square(pos_error), dim = -1)
    return penalty_error

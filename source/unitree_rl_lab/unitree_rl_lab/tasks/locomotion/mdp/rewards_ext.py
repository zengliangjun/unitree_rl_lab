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

    penalty = torch.norm(torch.abs(shoulder_pos - swing_target), dim = -1)

    return penalty

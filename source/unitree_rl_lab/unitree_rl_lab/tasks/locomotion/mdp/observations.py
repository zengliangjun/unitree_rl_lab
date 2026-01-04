from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.assets import Articulation

from . import cam_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def centroidal_angular_momentum_mixed(env: ManagerBasedRLEnv,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    M_com_bf = cam_utils.compute_com_mass_matrix_baseframe(asset, env)
    cmm_bf = cam_utils.get_cmm(M_com_bf)

    M_com_w = cam_utils.compute_com_mass_matrix_world_aligned(asset, env)
    cmm_w = cam_utils.get_cmm(M_com_w)

    qdot = cam_utils.get_generalized_qdot(asset)
    h_bf = cam_utils.compute_cam(cmm_bf, qdot)          # (B,6)
    h_w = cam_utils.compute_cam(cmm_w, qdot)          # (B,6)

    return torch.hstack([h_bf[:, :2], h_w[:, 2:3]])

def centroidal_angular_momentum_des_mixed(env: ManagerBasedRLEnv,
                command_name: str = "base_velocity",
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (B,3) = [vx,vy,wz]
    ndof = asset.num_joints

    M_com_bf = cam_utils.compute_com_mass_matrix_baseframe(asset, env)
    cmm_bf = cam_utils.get_cmm(M_com_bf)

    M_com_w = cam_utils.compute_com_mass_matrix_world_aligned(asset, env)
    cmm_w = cam_utils.get_cmm(M_com_w)

    h_ref_bf = cam_utils.compute_cam_ref(cmm_bf, cmd, ndof, env.device)
    h_ref_w = cam_utils.compute_cam_ref(cmm_w, cmd, ndof, env.device)

    return torch.hstack([h_ref_bf[:, :2], h_ref_w[:, 2:3]])


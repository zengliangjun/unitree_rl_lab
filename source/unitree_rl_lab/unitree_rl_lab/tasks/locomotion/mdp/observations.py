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
                command_name: str = "base_velocity",
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                sigma :float = 0.25) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (B,3) = [vx,vy,wz]
    ndof = asset.num_joints

    M_com = cam_utils.compute_com_mass_matrix(asset, env)
    cmm = cam_utils.get_cmm(M_com)

    qdot = cam_utils.get_generalized_qdot(asset)
    h = cam_utils.compute_cam(cmm, qdot)          # (B,6)
    h_ref = cam_utils.compute_cam_ref(cmm, cmd[:, :3], ndof, env.device)

    kz = h[:, 2]
    kz_hat = h_ref[:, 2]

    err = (kz_hat - kz) / (1.0 + torch.abs(kz_hat))
    r = torch.exp(- torch.square(err) / sigma)
    return r


class ArmCamDampingReward(ManagerTermBase):
    """r_dCAM in Eq.(9)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_cam_xy = torch.zeros(env.num_envs, 2, device=env.device)
        self.dt = env.step_dt

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.prev_cam_xy.zero_()
        else:
            self.prev_cam_xy[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

        asset: Articulation = env.scene[asset_cfg.name]
        M_com = cam_utils.compute_com_mass_matrix(asset, env)
        cmm = cam_utils.get_cmm(M_com)

        qdot = cam_utils.get_generalized_qdot(asset)
        h = cam_utils.compute_cam(cmm, qdot)  # (B,6)
        k_xy = h[:, 0:2]  # CAM x,y

        kdot_xy = (k_xy - self.prev_cam_xy) / self.dt
        self.prev_cam_xy = k_xy.detach()

        # Eq.(9): -min(0, sum_{i=x,y} k_i * kdot_i) :contentReference[oaicite:11]{index=11}
        s = torch.sum(k_xy * kdot_xy, dim=-1)
        r = -torch.minimum(torch.zeros_like(s), s)
        return r


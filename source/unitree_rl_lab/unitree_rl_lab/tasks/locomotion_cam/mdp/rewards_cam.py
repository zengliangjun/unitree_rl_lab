# reward_terms.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.assets import Articulation

from . import cam_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def armCamTrackingReward(env: ManagerBasedRLEnv,
                command_name: str = "base_velocity",
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                sigma :float = 0.25) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (B,3) = [vx,vy,wz]
    ndof = asset.num_joints

    M_com_w = cam_utils.compute_com_mass_matrix_world_aligned(asset, env)
    cmm_w = cam_utils.get_cmm(M_com_w)

    qdot = cam_utils.get_generalized_qdot(asset)
    h = cam_utils.compute_cam(cmm_w, qdot)          # (B,6)
    h_ref = cam_utils.compute_cam_ref(cmm_w, cmd[:, :3], ndof, env.device)

    kz = h[:, 2]
    kz_hat = h_ref[:, 2]

    err = (kz_hat - kz) / (1.0 + torch.abs(kz_hat))
    r = torch.exp(- torch.square(err * 2) / sigma)
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
        M_com = cam_utils.compute_com_mass_matrix_baseframe(asset, env)
        cmm = cam_utils.get_cmm(M_com)

        qdot = cam_utils.get_generalized_qdot(asset)
        h = cam_utils.compute_cam(cmm, qdot)  # (B,6)
        k_xy = h[:, 0:2]  # CAM x,y

        kdot_xy = (k_xy - self.prev_cam_xy) / self.dt
        self.prev_cam_xy = k_xy.detach()

        # Eq.(9): -min(0, sum_{i=x,y} k_i * kdot_i) :contentReference[oaicite:11]{index=11}
        s = torch.sum(k_xy * kdot_xy, dim=-1)
        r = -torch.clamp_min(torch.zeros_like(s), 0.0,)
        return r


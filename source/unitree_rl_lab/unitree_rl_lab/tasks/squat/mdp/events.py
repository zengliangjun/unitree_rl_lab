
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs import ManagerBasedEnv

from unitree_rl_lab.tasks.locomotion import mdp

def push_by_setting_velocity_with_level(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    max_velocity_range: dict[str, tuple[float, float]],
    speed: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):

    mdp.push_by_setting_velocity(
            env,
            env_ids,
            velocity_range,
            asset_cfg)

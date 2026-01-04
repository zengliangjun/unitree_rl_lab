# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import Sequence, TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg, ManagerTermBase
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BaseTermination(ManagerTermBase):

    _env: ManagerBasedRLEnv

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.terminated_buffer = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        max_lin_vel: float = None,
        max_ang_vel: float = None,
        max_tilting: float = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Terminate when the asset's linear velocity exceeds the maximum linear velocity."""
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if max_lin_vel is not None:
            terminated |= torch.any(torch.norm(asset.data.root_lin_vel_b, dim=-1, keepdim=True) > max_lin_vel, dim=1)
        if max_ang_vel is not None:
            terminated |= torch.any(torch.norm(asset.data.root_ang_vel_b, dim=-1, keepdim=True) > max_ang_vel, dim=1)
        if max_tilting is not None:
            terminated |= torch.any(torch.abs(asset.data.projected_gravity_b[:, 0:1]) > max_tilting, dim=1)
            terminated |= torch.any(torch.abs(asset.data.projected_gravity_b[:, 1:2]) > max_tilting, dim=1)

        self.terminated_buffer[...] = terminated
        return terminated

"""
Contact sensor.
"""
class IllegalContact(ManagerTermBase):

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        asset: Articulation = env.scene[asset_cfg.name]
        self.upper_ids = asset.find_bodies(cfg.params["upper_names"])[0]
        self.leg_ids = asset.find_bodies(cfg.params["leg_names"])[0]

        self.upper_terminated_buffer = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.leg_terminated_buffer = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float = 1,
        upper_names: Sequence[str] = None,
        leg_names: Sequence[str] = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    ) -> torch.Tensor:

        """Terminate when the contact force on the sensor exceeds the force threshold."""
        # extract the used quantities (to enable type-hinting)
        contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
        net_contact_forces = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
        upper_terminated = torch.norm(net_contact_forces[:, self.upper_ids], dim=-1) > threshold
        leg_terminated = torch.norm(net_contact_forces[:, self.leg_ids], dim=-1) > threshold

        self.upper_terminated_buffer[...] = upper_terminated
        self.leg_terminated_buffer[...] = leg_terminated

        return upper_terminated | leg_terminated

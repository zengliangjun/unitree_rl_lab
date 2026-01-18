
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def reset_joints_by_range(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by sampling random values from the given ranges.

    This function samples random values from the given ranges and sets them into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids, asset_cfg.joint_ids].clone()

    position_range = torch.tensor(position_range, device=asset.device)
    velocity_range = torch.tensor(velocity_range, device=asset.device)

    # get default joint state
    joint_pos = joint_pos + math_utils.sample_uniform(position_range[..., 0], position_range[..., 1], (len(env_ids), asset.num_joints), device=asset.device)
    joint_vel = joint_vel + math_utils.sample_uniform(velocity_range[..., 0], velocity_range[..., 1], (len(env_ids), asset.num_joints), device=asset.device)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def apply_external_force_torque_disturbance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
):
    """Apply the random external force and torque to the robot once per call.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=asset.device)

    body_ids = asset_cfg.body_ids

    # resolve number of bodies
    num_bodies = len(body_ids) if isinstance(body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device) / env.physics_dt
    torques = math_utils.sample_uniform(*torque_range, size, asset.device) / env.physics_dt

    _external_force_b = torch.zeros((env.scene.num_envs, asset.num_bodies, 3), device=env.device)
    _external_torque_b = torch.zeros_like(_external_force_b)

    if isinstance(body_ids, list):
        body_ids = torch.tensor(body_ids, dtype=torch.long, device=env.device)
    else:
        body_ids = torch.arange(num_bodies, device=asset.device)

    indices = body_ids.repeat(len(env_ids), 1) + \
              env_ids.unsqueeze(1) * num_bodies
    indices = indices.view(-1)

    _external_force_b.flatten(0, 1)[indices] = forces.flatten(0, 1)
    _external_torque_b.flatten(0, 1)[indices] = torques.flatten(0, 1)

    asset.root_physx_view.apply_forces_and_torques_at_position(
        force_data=_external_force_b.view(-1, 3),
        torque_data=_external_torque_b.view(-1, 3),
        position_data=None,
        indices=env_ids,
        is_global=False,
    )


def apply_force_mix(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
):

    shuffled_indices = torch.randperm(env_ids.shape[0], device=env.device)  # 打乱索引
    velocity_len = env_ids.shape[0] // 2

    velocity_indices = shuffled_indices[:velocity_len]
    external_indices = shuffled_indices[velocity_len:]

    velocity_env_ids = env_ids[velocity_indices]
    external_env_ids = env_ids[external_indices]

    from isaaclab.envs.mdp import events
    if 0 != velocity_env_ids.shape[0]:
        events.push_by_setting_velocity(env, velocity_env_ids, velocity_range, asset_cfg)

    if 0 != external_env_ids.shape[0]:
        apply_external_force_torque_disturbance(env, external_env_ids, force_range, torque_range, asset_cfg)

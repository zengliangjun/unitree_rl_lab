
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from typing import Literal

from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject

from unitree_rl_lab.tasks.locomotion import mdp
from . import command_squat


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


def apply_support_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    coefficient: float = 0.3,
    min_coefficient: float = 0,
    max_coefficient: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):

    if coefficient <= min_coefficient:
        mdp.apply_external_force_torque(
        env,
        env_ids,
        force_range,
        torque_range,
        asset_cfg)
        return

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = torch.ones(size, dtype=torch.float32, device=asset.device)
    torques = torch.zeros_like(forces)

    forces[:, :, :2] = 0

    mass = asset.data.default_mass.sum(dim=-1, keepdim=True).to(env.device)


    invert_forces = mass[env_ids] * 9.8 * coefficient
    forces[:, :, 2] = invert_forces
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)



def apply_support_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    recompute_inertia: bool = True,

    coefficient: float = 0.3,
    min_coefficient: float = 0.2,
    max_coefficient: float = 1,

):

    if coefficient >= max_coefficient:
        mdp.randomize_rigid_body_mass(
                env,
                env_ids,
                asset_cfg,
                mass_distribution_params,
                operation,
                distribution,
                recompute_inertia
                )
        return


    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current masses of the bodies (num_assets, num_bodies)
    masses = asset.root_physx_view.get_masses()

    # apply randomization on default values
    # this is to make sure when calling the function multiple times, the randomization is applied on the
    # default values and not the previously randomized values
    masses[env_ids[:, None], body_ids] = asset.data.default_mass[env_ids[:, None], body_ids].clone()

    # sample from the given range
    # note: we modify the masses in-place for all environments
    #   however, the setter takes care that only the masses of the specified environments are modified
    masses *= coefficient

    # set the mass into the physics simulation
    asset.root_physx_view.set_masses(masses, env_ids)

    # recompute inertia tensors if needed
    if recompute_inertia:
        # compute the ratios of the new masses to the initial masses
        ratios = masses[env_ids[:, None], body_ids] / asset.data.default_mass[env_ids[:, None], body_ids]
        # scale the inertia tensors by the the ratios
        # since mass randomization is done on default values, we can use the default inertia tensors
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            # inertia has shape: (num_envs, num_bodies, 9) for articulation
            inertias[env_ids[:, None], body_ids] = (
                asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            )
        else:
            # inertia has shape: (num_envs, 9) for rigid object
            inertias[env_ids] = asset.data.default_inertia[env_ids] * ratios
        # set the inertia tensors into the physics simulation
        asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_rigid_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    recompute_inertia: bool = True,
    command_term_name: str = "squat_command",
):

    command_term: command_squat.SquatCommand = env.command_manager.get_term(command_term_name)

    if command_term.average_episode_length > env.max_episode_length * 0.96:
        mdp.randomize_rigid_body_mass(
            env,
            env_ids,
            asset_cfg,
            mass_distribution_params,
            operation,
            distribution,
            recompute_inertia
        )

def apply_external_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_term_name: str = "squat_command",
):

    command_term: command_squat.SquatCommand = env.command_manager.get_term(command_term_name)
    if command_term.average_episode_length > env.max_episode_length * 0.96:
        mdp.apply_external_force_torque(
            env,
            env_ids,
            force_range,
            torque_range,
            asset_cfg,
        )

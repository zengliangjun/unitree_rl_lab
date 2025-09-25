from dataclasses import MISSING
try:
    from isaaclab.utils import configclass
except:
    from configclasses import configclass

import torch

@configclass
class IsaacSimConfig:
    joint_names: list[str] = MISSING
    actuator_stiffness: list[float] | torch.Tensor = MISSING
    actuator_damping: list[float] | torch.Tensor = MISSING
    init_pos: list[float] | torch.Tensor = MISSING

    ang_vel_objsscale: list[float] | torch.Tensor = MISSING
    gravity_objsscale: list[float] | torch.Tensor = MISSING
    commands_objsscale: list[float] | torch.Tensor = MISSING
    joint_pos_objsscale: list[float] | torch.Tensor = MISSING
    joint_vel_objsscale: list[float] | torch.Tensor = MISSING
    action_objsscale: list[float] | torch.Tensor = MISSING

    observations_names: list[str] = MISSING
    observations_length: int = MISSING

    simulation_dt: float = MISSING
    decimation: int = MISSING
    action_scale: float = MISSING

    '''
    set by init_configs
    '''
    deploy_objsscales: torch.Tensor = None


@configclass
class MujocoConfig:
    sim_config : IsaacSimConfig = MISSING

    policy_path: str = MISSING
    mujoco_path: str = MISSING

    simulation_duration: float = MISSING

    command: list[float] | torch.Tensor = MISSING


def init_configs(cfg: MujocoConfig):

    sim_config = cfg.sim_config
    all_objsscales = []
    for name in sim_config.observations_names:

        scales = getattr(sim_config, f"{name}_objsscale")
        if not isinstance(scales, torch.Tensor):
            scales = torch.tensor(scales)
        all_objsscales.append(scales)

    all_objsscales = torch.concat(all_objsscales)
    sim_config.deploy_objsscales = all_objsscales

    if not isinstance(sim_config.actuator_stiffness, torch.Tensor):
        sim_config.actuator_stiffness = torch.tensor(sim_config.actuator_stiffness)

    if not isinstance(sim_config.actuator_damping, torch.Tensor):
        sim_config.actuator_damping = torch.tensor(sim_config.actuator_damping)

    if not isinstance(sim_config.init_pos, torch.Tensor):
        sim_config.init_pos = torch.tensor(sim_config.init_pos)

    if not isinstance(cfg.command, torch.Tensor):
        cfg.command = torch.tensor(cfg.command)

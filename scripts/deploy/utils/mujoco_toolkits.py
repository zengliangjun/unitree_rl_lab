from dataclasses import MISSING
try:
    from isaaclab.utils import configclass
except:
    from configclasses import configclass

import torch
import mujoco.viewer
import mujoco
import numpy as np
import time

from configs import configs
from utils.context import MujocoContext

def init_context(cfg: configs.MujocoConfig) -> MujocoContext:

    module = mujoco.MjModel.from_xml_path(cfg.mujoco_path)
    data = mujoco.MjData(module)
    module.opt.timestep = cfg.sim_config.simulation_dt

    ##
    maps_mujoco2isaac_sim = []
    for name in cfg.sim_config.joint_names:
        mujocoid = mujoco.mj_name2id(module, mujoco.mjtObj.mjOBJ_JOINT, name)
        print("mujoco2isaac_sim", name, mujocoid)
        maps_mujoco2isaac_sim.append(mujocoid -1)

    maps_isaac_sim2mujoco = []
    for mujocoid in range(1, len(maps_mujoco2isaac_sim) + 1):
        joint_name = mujoco.mj_id2name(module, mujoco.mjtObj.mjOBJ_JOINT, mujocoid)
        simid = cfg.sim_config.joint_names.index(joint_name)
        print("isaac_sim2mujoco", mujocoid, simid)

        maps_isaac_sim2mujoco.append(simid)

    return MujocoContext(
            mujoco_module = module,
            mujoco_data = data,

            maps_mujoco2isaac_sim = maps_mujoco2isaac_sim,
            maps_isaac_sim2mujoco = maps_isaac_sim2mujoco,
        )

def init_viewer(context: MujocoContext):
    return mujoco.viewer.launch_passive(context.mujoco_module, context.mujoco_data)


def sleep(ctxt: MujocoContext, step_start):
    time_until_next_step = ctxt.mujoco_module.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)


def step(context: MujocoContext, sim_torques_numpy: np.ndarray):
    mujoco_torques = sim_torques_numpy[context.maps_isaac_sim2mujoco]

    context.mujoco_data.ctrl[:] = mujoco_torques
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(context.mujoco_module, context.mujoco_data)

def dofStatus2simTensor(context: MujocoContext) -> tuple:
    pos = context.mujoco_data.qpos[7:]
    vel = context.mujoco_data.qvel[6:]

    pos = pos[context.maps_mujoco2isaac_sim]
    vel = vel[context.maps_mujoco2isaac_sim]

    return (torch.Tensor(pos), torch.Tensor(vel))


def objs2simTensor(context: MujocoContext, quaternion2gravity) -> dict:

    quat = context.mujoco_data.qpos[3:7]

    gravity = quaternion2gravity(quat)
    joint_pos = context.mujoco_data.qpos[7:]

    ang_vel = context.mujoco_data.qvel[3:6]
    joint_vel = context.mujoco_data.qvel[6:]


    joint_pos = joint_pos[context.maps_mujoco2isaac_sim]
    joint_vel = joint_vel[context.maps_mujoco2isaac_sim]

    return {
        "gravity": torch.Tensor(gravity),
        "ang_vel": torch.Tensor(ang_vel),
        "joint_pos": torch.Tensor(joint_pos),
        "joint_vel": torch.Tensor(joint_vel),
    }


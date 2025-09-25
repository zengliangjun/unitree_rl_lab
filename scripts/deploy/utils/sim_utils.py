import torch
from configs import configs
from utils.context import MujocoContext
import numpy as np

def quaternion2gravity(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pdTensor2Numpy(simcfg: configs.IsaacSimConfig, simstatus_tensors: tuple, sim_target_pos: torch.Tensor = None):

    pos, vel = simstatus_tensors
    if sim_target_pos is None:
        sim_target_pos = torch.tensor(simcfg.init_pos)

    torques: torch.Tensor = (sim_target_pos - pos) * simcfg.actuator_stiffness + \
              (0 - vel) * simcfg.actuator_damping

    return torques.detach().numpy()

def step2objs(sim_cfg: configs.IsaacSimConfig, simObjTensors: dict):
    simObjTensors['joint_pos'] -= sim_cfg.init_pos

    objs = []
    for name in sim_cfg.observations_names:
        try:
            items = simObjTensors[name]
        except:
            items = getattr(simObjTensors, name)
        if not isinstance(items, torch.Tensor):
            items = torch.tensor(items)
        objs.append(items)

    objs = torch.cat(objs)
    objs = objs.unsqueeze(0) * sim_cfg.deploy_objsscales
    return objs

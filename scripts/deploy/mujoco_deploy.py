import os.path as osp
import math
import sys
import time
import torch

root = osp.join(osp.dirname(__file__))
sys.path.insert(0, root)

from configs import configs
from utils import mujoco_toolkits, policy_toolkits, sim_utils, context

def main(cfg: configs.MujocoConfig):

    sim_cfg: configs.IsaacSimConfig = cfg.sim_config
    ctxt: context.MujocoContext = mujoco_toolkits.init_context(cfg)
    policy = policy_toolkits.load(cfg)

    counter = 0
    simpos_tensors: torch.Tensor = None

    action = torch.zeros_like(sim_cfg.init_pos)
    history_objs: torch.Tensor = None
    with mujoco_toolkits.init_viewer(ctxt) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        #while viewer.is_running() and time.time() - start < cfg.simulation_duration:
        while viewer.is_running():
            step_start = time.time()

            simstatus_tensors = mujoco_toolkits.dofStatus2simTensor(ctxt)
            sim_torques_numpy = sim_utils.pdTensor2Numpy(sim_cfg, simstatus_tensors, simpos_tensors)
            mujoco_toolkits.step(ctxt, sim_torques_numpy)

            counter += 1
            if counter % sim_cfg.decimation == 0:
                simObjTensors: dict = mujoco_toolkits.objs2simTensor(ctxt, sim_utils.quaternion2gravity)
                simObjTensors["commands"] = cfg.command
                simObjTensors["action"] = action

                if "phase" in sim_cfg.observations_names:
                    period = 0.8
                    count = counter * sim_cfg.simulation_dt
                    phase = count % period / period
                    angle = 2 * math.pi * phase

                    phase = torch.tensor([math.sin(angle), math.cos(angle)], dtype = torch.float32)
                    simObjTensors["phase"] = phase

                objs_tensors = sim_utils.step2objs(sim_cfg, simObjTensors)

                # policy inference
                objs_size = objs_tensors.shape[1]
                imput_size = objs_size * sim_cfg.observations_length
                if history_objs is None:
                    history_objs = torch.zeros((objs_tensors.shape[0], \
                                               imput_size),\
                                               dtype = objs_tensors.dtype)
                else:
                    history_objs[:, :-objs_size] = history_objs[:, objs_size:].clone()

                history_objs[:, -objs_size:] = objs_tensors

                action = policy(history_objs)[0].detach()

                # transform action to target_dof_pos
                simpos_tensors = action * sim_cfg.action_scale + sim_cfg.init_pos

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            mujoco_toolkits.sleep(ctxt, step_start)

if __name__ == "__main__":
    if True:
        from configs import configs_g129dof
        cfg = configs_g129dof.G129dofConfig()
    else:
        from configs import configs_g112dof
        cfg = configs_g112dof.G112dofConfig()
    configs.init_configs(cfg)
    main(cfg)

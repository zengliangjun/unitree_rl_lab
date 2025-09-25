from dataclasses import MISSING
try:
    from isaaclab.utils import configclass
except:
    from configclasses import configclass


@configclass
class MujocoContext:

    mujoco_module = None
    mujoco_data = None

    maps_mujoco2isaac_sim = None
    maps_isaac_sim2mujoco = None


from isaaclab.utils import configclass
from unitree_rl_lab.assets.robots.lyenbot_legs_kp160 import LYENBOT_CFG as ROBOT_CFG

from . import stomp_kp125_env_cfg

@configclass
class RobotEnvCfg(stomp_kp125_env_cfg.RobotEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



@configclass
class RobotPlayEnvCfg(stomp_kp125_env_cfg.RobotPlayEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



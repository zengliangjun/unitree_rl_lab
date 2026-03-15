from . import velocity_env_cfg
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotionv2.mdp import rewards_ext

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

@configclass
class Robot30EnvCfg(velocity_env_cfg.Robot30EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.penalty_feet_orientation.weight = -3.0
        self.rewards.reward_feet_clearance.weight = 0.5
        self.rewards.penalize_feet_forces.weight = -0.03
        self.rewards.reward_feet_clearance.func = rewards_ext.reward_foot_clearance_v2
        self.rewards.penalize_feet_clearance.func = rewards_ext.penalize_foot_clearance_v2

@configclass
class Robot30PlayEnvCfg(velocity_env_cfg.Robot30PlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()



@configclass
class Robot30Env2Cfg(Robot30EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.penalty_feet_orientation.weight = -3.0
        self.rewards.reward_feet_clearance.weight = 0.5
        self.rewards.penalize_feet_forces.weight = -0.03
        self.rewards.reward_feet_clearance.func = rewards_ext.reward_foot_clearance_v2
        self.rewards.penalize_feet_clearance.func = rewards_ext.penalize_foot_clearance_v2

        self.rewards.reward_feet_clearance.params["target_height"] = 0.08
        self.rewards.penalize_feet_clearance.params["target_height"] = 0.08

        self.rewards.feet_width = RewTerm(
        func=rewards_ext.reward_feet_width,
        weight=0.5,
        params={
            "asset_cfg":
            SceneEntityCfg("robot", body_names=[
                "left_knee_pitch_link",
                "right_knee_pitch_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link"],
            preserve_order=True),
            "target_width": 0.275, # 0.32, # 0.23
            "std": 0.12
        },
    )

@configclass
class Robot30PlayEnv2Cfg(Robot30PlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()

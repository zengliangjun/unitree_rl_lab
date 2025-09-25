import math


from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.assets.terrains.config import rough_low_level_cfg

from . import velocity_env_cfg

@configclass
class RewardsCfg(velocity_env_cfg.RewardsCfg):

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )

    joint_deviation_arms_2 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )
    bodies_symmetry = RewTerm(
        func=mdp.BodiesSymmetry,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    ".*left_elbow_link",
                    ".*right_elbow_link",
                    ".*left_wrist_yaw_link",
                    ".*right_wrist_yaw_link"
                ],
                preserve_order=True
            ),
            "command_name": "base_velocity",
            "std_ranges": [0.08, 0.21],
            "error_std": 0.02
        },
    )


@configclass
class RobotEnvCfg(velocity_env_cfg.RobotEnvCfg):
    rewards: RewardsCfg = RewardsCfg()

@configclass
class RobotPlayEnvCfg(velocity_env_cfg.RobotPlayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RewardsTimeCfg(RewardsCfg):
    rew_steps = RewTerm(
        func=mdp.TimesSymmetry,
        weight=0.1,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    ".*left_ankle_roll_link",
                    ".*right_ankle_roll_link"
                ]
            ),
            "error_std": 0.1
        },
    )

@configclass
class RobotStepsCfg(RobotEnvCfg):
    rewards: RewardsTimeCfg = RewardsTimeCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0, 1.0)

@configclass
class RobotStepsPlayCfg(RobotPlayEnvCfg):
    rewards: RewardsTimeCfg = RewardsTimeCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0, 1.0)


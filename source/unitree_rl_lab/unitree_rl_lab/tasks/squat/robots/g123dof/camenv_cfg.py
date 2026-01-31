
from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import squat_env_cfg
from unitree_rl_lab.tasks.squat.mdp import rewards_cam
from unitree_rl_lab.tasks.squat.mdp import rewards
import math

@configclass
class RewardsCfg(squat_env_cfg.RewardsCfg):

    dCAM_xy = RewTerm(
        func=rewards_cam.ArmCamDampingReward,
        weight=-5e-4,
        params={"command_name": "squat_command",
                "asset_cfg": SceneEntityCfg("robot")}
    )
    tracking_CAM_reward = RewTerm(
        func=rewards_cam.armCamTrackingReward,
        weight=5.1,
        params={"command_name": "squat_command",
                "asset_cfg": SceneEntityCfg("robot")}
    )

    penalty_squat_pos = RewTerm(
        func=rewards.track_squat_error,
        weight=- 1e-3,
        params={"command_name": "squat_command",
                "finished_weight": 1.6,
                "finished_max_weight": 2.4,
                "penalty_weight": 1,
                "penalty_max_weight": 1.5,
                "std": math.sqrt(0.01),
                "asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_knee_joint",
                        "right_knee_joint"],
                    preserve_order=True)}
    )

    penalty_lin_z = RewTerm(
        func=rewards.penalty_lin_vel_z_v1,
        weight= - 10,
        params={"command_name": "squat_command",
                "asset_cfg": SceneEntityCfg("robot")}
    )

    def __post_init__(self):
        self.track_symmetry_pos.weight = 0.25
        self.com_zero.weight = 0.25
        # self.zero_ang_vel.weight = 0.15
        # self.zero_lin_xy_vel.weight = None

@configclass
class G123EnvCfg(squat_env_cfg.RobotEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.rewards: RewardsCfg = RewardsCfg()
        self.observations.policy.history_length = 10
        self.observations.critic.history_length = 10
        self.rewards.reward_pitch2zero.params["std"] = 0.36


@configclass
class G123PlayEnvCfg(squat_env_cfg.RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.rewards: RewardsCfg = RewardsCfg()
        self.observations.policy.history_length = 10
        self.observations.critic.history_length = 10


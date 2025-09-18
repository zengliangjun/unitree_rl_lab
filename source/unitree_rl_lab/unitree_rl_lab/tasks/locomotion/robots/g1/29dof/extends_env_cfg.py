import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

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

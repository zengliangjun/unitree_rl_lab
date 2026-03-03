
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from . import unitree

LYENBOT_CFG = unitree.UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{unitree.UNITREE_MODEL_DIR}/lyenbot/usd_lyenbotleg/lyenbotleg_20151206_collision.usd",
        usd_path=f"{unitree.UNITREE_MODEL_DIR}/lyenbot/usd_lyenbotleg_fix_ankle/lyenbotlegs_20260212_collision_ankle_roll_pitch.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.90),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_pitch_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"],  # 5
            effort_limit_sim={
                ".*_hip_pitch_joint": 140,
                ".*_hip_yaw_joint": 55,
                "waist_yaw_joint": 55
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 12.04,
                ".*_hip_yaw_joint": 17.79,
                "waist_yaw_joint": 17.7
            },
            stiffness={
                ".*_hip_pitch_joint": 100,  #
                ".*_hip_yaw_joint": 50,  #
                "waist_yaw_joint": 75,  #
            },
            damping={
                ".*_hip_pitch_joint": 4,
                ".*_hip_yaw_joint": 2,
                "waist_yaw_joint": 3
            },
            armature=0.010177520,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_knee_pitch_joint"],  # 4
            effort_limit_sim={
                ".*_hip_roll_joint": 55,
                ".*_knee_pitch_joint": 140
            },
            velocity_limit_sim={
                ".*_hip_roll_joint": 17.79,
                ".*_knee_pitch_joint": 12.04
            },
            stiffness={
                ".*_hip_roll_joint": 75,  #
                ".*_knee_pitch_joint": 100,  #
            },
            damping={
                ".*_hip_roll_joint": 3,
                ".*_knee_pitch_joint": 4
            },
            armature=0.025101925,
        ),
        "N5020-16-parallel": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle.*"],  # 4
            effort_limit_sim={
                ".*ankle_pitch.*": 66,
                ".*ankle_roll.*": 33
            },
            velocity_limit_sim=16.22,
            stiffness=20,
            damping=2.65,
            armature=0.007219450,
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_pitch_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",

        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_pitch_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",

        "waist_yaw_joint",

    ],
    legs_joint_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_pitch_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_pitch_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint"
    ],
    arms_joint_names=[
    ],
    left_knee_name="left_knee_pitch_joint",
    right_knee_name="right_knee_pitch_joint"
)

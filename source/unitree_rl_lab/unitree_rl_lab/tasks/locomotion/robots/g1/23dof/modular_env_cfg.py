# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from unitree_rl_lab.tasks.locomotion import mdp  # noqa: F401, F403
from unitree_rl_lab.tasks.locomotion.mdp import commands_ext_cfg

##
# Pre-defined configs
##
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG
from unitree_rl_lab.terrains import ROUGH_TERRAINS_CFG

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane", # "plane" or "generator"
        terrain_generator=ROUGH_TERRAINS_CFG, # ROUGH_TERRAINS_CFG, COBBLESTONE_ROAD_CFG
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            # static_friction=1.0,
            # dynamic_friction=1.0,
            static_friction=0.8,
            dynamic_friction=0.8,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = UNITREE_G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    # # camera
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = commands_ext_cfg.CommandExtCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),

        rel_standing_envs=0.1,
        rel_walking_envs= 0.23,
        rel_spining_envs= 0.23,
        rel_walking_spining_envs= 0.44,

        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=commands_ext_cfg.CommandExtCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=commands_ext_cfg.CommandExtCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-0.8, 0.8)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    leg_joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=UNITREE_G1_23DOF_CFG.legs_joint_names) #! The order of the joints are not correct.
    arm_joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=UNITREE_G1_23DOF_CFG.arms_joint_names) #! The order of the joints are not correct.
    # arm_joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=HUMANOID_FULL_CFG.actuators['arms'].joint_names_expr, disable_action=True) #! The order of the joints are not correct.


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ActorCfg(ObsGroup):
        """Observations for leg actor. (order preserved)"""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(std=0.05))
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Gnoise(std=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Gnoise(std=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        legs_pos = ObsTerm(func=mdp.joint_pos, noise=Gnoise(std=0.05),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.legs_joint_names)})
        legs_vel = ObsTerm(func=mdp.joint_vel, noise=Gnoise(std=1.0),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.legs_joint_names)})
        legs_action = ObsTerm(func=mdp.last_action, params={"action_name": "leg_joint_pos"}, noise=Gnoise(std=0.1))

        arms_pos = ObsTerm(func=mdp.joint_pos, noise=Gnoise(std=0.05),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.arms_joint_names)})
        arms_vel = ObsTerm(func=mdp.joint_vel, noise=Gnoise(std=1.0),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.arms_joint_names)})
        arms_action = ObsTerm(func=mdp.last_action, params={"action_name": "arm_joint_pos"}, noise=Gnoise(std=0.1))

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for leg critic. (order preserved)"""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        legs_pos = ObsTerm(func=mdp.joint_pos, noise=Gnoise(std=0.05),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.legs_joint_names)})
        legs_vel = ObsTerm(func=mdp.joint_vel, noise=Gnoise(std=1.0),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.legs_joint_names)})
        legs_action = ObsTerm(func=mdp.last_action, params={"action_name": "leg_joint_pos"}, noise=Gnoise(std=0.1))

        arms_pos = ObsTerm(func=mdp.joint_pos, noise=Gnoise(std=0.05),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.arms_joint_names)})
        arms_vel = ObsTerm(func=mdp.joint_vel, noise=Gnoise(std=1.0),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.arms_joint_names)})
        arms_action = ObsTerm(func=mdp.last_action, params={"action_name": "arm_joint_pos"}, noise=Gnoise(std=0.1))

        CAM = ObsTerm(func=mdp.centroidal_angular_momentum_mixed, noise=Gnoise(std=0.1))
        CAM_des = ObsTerm(func=mdp.centroidal_angular_momentum_des_mixed)



        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    leg_actor: ActorCfg = ActorCfg()
    leg_critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for randomization by events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x":     (-0.5, 0.5),
                "y":     (-0.5, 0.5),
                "z":     (-3.14, 3.14),
                "roll":  (-torch.pi/20, torch.pi/20),
                "pitch": (-torch.pi/20, torch.pi/20),
                "yaw":   (-torch.pi, torch.pi)
            },
            "velocity_range": {
                "x":     (-.1, .1),
                "y":     (-.1, .1),
                "z":     (-.1, .1),
                "roll":  (-.1, .1),
                "pitch": (-.1, .1),
                "yaw":   (-.1, .1),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_range,
        mode="reset",
        params={
            "position_range": torch.tensor([[-0.1, 0.1]]),
            "velocity_range": torch.tensor([[-0.1, 0.1]]),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_xy_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class EventsDeployCfg:
    """Configuration for randomization by events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x":     (-0., 0.),
                    "y":     (-0., 0.),
                    "z":     (-0., 0.),
                    "roll":  (-0., 0.),
                    "pitch": (-0., 0.),
                    "yaw":   (-torch.pi, torch.pi)
                },
                "velocity_range": {
                    "x":     (-0., 0.),
                    "y":     (-0., 0.),
                    "z":     (-0., 0.),
                    "roll":  (-0., 0.),
                    "pitch": (-0., 0.),
                    "yaw":   (-0., 0.),
                },
            },
        )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # interval
    apply_external_force = EventTerm(
        func=mdp.apply_external_force_torque_disturbance,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"force_range": (-15.0, 15.0), "torque_range": (-1.5, 1.5)},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    @configclass
    class LegRewardsCfg:
        """Reward terms for the leg actor-critic."""

        base_lin_vel_z = RewTerm(
            func=mdp.lin_vel_z_l2,
            weight=- 1e-1,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        base_ang_vel_xy = RewTerm(
            func=mdp.ang_vel_xy_l2,
            weight=- 1e-2,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        # * Regularization rewards * #
        action_smoothness1 = RewTerm(
            func=mdp.action_smoothness1,
            weight=- 2e-3,
            params={"action_name": "leg_joint_pos"}
        )
        action_smoothness2 = RewTerm(
            func=mdp.action_smoothness2,
            weight=- 2e-4,
            params={"action_name": "leg_joint_pos"}
        )
        joint_torque = RewTerm(
            func=mdp.joint_torques_l2,
            weight=- 1e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.legs_joint_names)}
        )
        joint_velocity = RewTerm(
            func=mdp.joint_vel_l2,
            weight=- 2e-3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.legs_joint_names)}
        )
        joint_pos_limits = RewTerm(
            func=mdp.joint_pos_limits,
            weight=- 10,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.legs_joint_names)}
        )
        joint_torque_limits = RewTerm(
            func=mdp.applied_torque_limits,
            weight=- 1e-2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.legs_joint_names)}
        )
        joint_regularization = RewTerm(
            func=mdp.joint_regularization,
            weight=1.,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                                                            "left_hip_roll_joint",
                                                            "left_hip_yaw_joint",
                                                            "right_hip_roll_joint",
                                                            "right_hip_yaw_joint",
                                                            "waist_yaw_joint"
                                                        ])}
        )

        # * Floating base rewards * #
        # base_height = RewTerm(
        #     func=brl_mdp.base_height_reward,
        #     weight=1.0,
        #     params={"asset_cfg": SceneEntityCfg("robot"), "base_height_target": 0.62}
        # )
        # base_heading = RewTerm(
        #     func=brl_mdp.base_heading_reward,
        #     weight=3.0,
        #     params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"}
        # )
        base_z_orientation = RewTerm(
            func=mdp.orientation_reward,
            weight=0.7, # 1.0,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        # tracking_lin_vel_world = RewTerm(
        #     func=brl_mdp.tracking_lin_vel_world_reward,
        #     weight=4.0,
        #     params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"}
        # )
        tracking_lin_vel_xy = RewTerm(
            func=mdp.track_lin_vel_reward,
            weight=4.6, # 4.0,
            params={"asset_cfg": SceneEntityCfg("robot"),
                    "command_name": "base_velocity",
                    "std": math.sqrt(0.25)}
        )
        tracking_yaw_vel = RewTerm(
            func=mdp.track_ang_vel_reward,
            weight=2.51, #1.0,
            params={"asset_cfg": SceneEntityCfg("robot"),
                    "command_name": "base_velocity",
                    "std": math.sqrt(0.25)}
        )
        # * Stepping rewards * #
        gait = RewTerm(
            func=mdp.feet_gait,
            weight=3,
            params={
                "period": 0.8,
                "offset": [0.0, 0.5],
                "threshold": 0.55,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            },
        )

        # * Termination rewards * #
        termination = RewTerm(
            func=brl_mdp.termination_penalty,
            weight=1.0,
            params={"group_name": "leg"}
        )

    @configclass
    class ArmRewardsCfg:
        """Reward terms for the arm actor-critic."""
        # * Regularization rewards * #
        action_smoothness1 = RewTerm(
            func=mdp.action_smoothness1,
            weight=- 1e-3,
            params={"action_name": "arm_joint_pos"}
        )
        action_smoothness2 = RewTerm(
            func=mdp.action_smoothness2,
            weight=- 1e-4,
            params={"action_name": "arm_joint_pos"}
        )
        joint_torque = RewTerm(
            func=mdp.joint_torques_l2,
            weight=- 5e-3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.arms_joint_names)}
        )
        joint_velocity = RewTerm(
            func=mdp.joint_vel_l2,
            weight=- 5e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.arms_joint_names)}
        )
        joint_position = RewTerm(
            func=mdp.joint_position_penalty,
            weight=- 1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.arms_joint_names)}
        )
        joint_pos_limits = RewTerm(
            func=mdp.joint_pos_limits,
            weight=- 10,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.arms_joint_names)}
        )
        joint_torque_limits = RewTerm(
            func=mdp.applied_torque_limits,
            weight=- 1e-2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_23DOF_CFG.arms_joint_names)}
        )
        dCAM_xy = RewTerm(
            func=mdp.dCAM_xy_penalty,
            weight=5e-2,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        tracking_CAM_reward = RewTerm(
            func=mdp.tracking_CAM_reward,
            weight=3.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"}
        )

        # * Termination rewards * #
        termination = RewTerm(
            func=mdp.termination_penalty,
            weight=1.0,
            params={"group_name": "arm"}
        )

    leg: LegRewardsCfg = LegRewardsCfg()
    arm: ArmRewardsCfg = ArmRewardsCfg()


@configclass
class TerminationsCfg:

    illegal_contact = DoneTerm(
        func=mdp.IllegalContact,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 1.0,
            "upper_names": [
                'left_shoulder_pitch_link',
                'left_shoulder_roll_link',
                'left_shoulder_yaw_link',
                'left_elbow_link',
                'left_wrist_roll_link',
                'left_wrist_pitch_link',
                'left_wrist_yaw_link',
                'left_rubber_hand',

                'right_shoulder_pitch_link',
                'right_shoulder_roll_link',
                'right_shoulder_yaw_link',
                'right_elbow_link',
                'right_wrist_roll_link',
                'right_wrist_pitch_link',
                'right_wrist_yaw_link',
                'right_rubber_hand'
            ],
            "leg_names": [
                'left_hip_pitch_link',
                'left_hip_roll_link',
                'left_hip_yaw_link',
                'left_knee_link',

                'right_hip_pitch_link',
                'right_hip_roll_link',
                'right_hip_yaw_link',
                'right_knee_link',

                'waist_yaw_link',
                'waist_roll_link',
                'torso_link',
            ]
            },
    )

    base_termination = DoneTerm(
        func=mdp.BaseTermination,
        params={
            "max_lin_vel": 15.0,
            "max_ang_vel": 10.0,
            "max_tilting": 0.8,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class HumanoidFullModularEnvCfg(ManagerBasedRLEnvCfg):
    viewer = ViewerCfg(eye=(2.0, -2.0, 0.5), origin_type='asset_root', asset_name='robot')
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3.)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsDeployCfg = EventsDeployCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 10
        # video recording settings
        self.video_length_s = 3
        # simulation settings
        self.sim.dt = 0.005
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if hasattr(self.scene, 'height_scanner'):
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if hasattr(self.scene, 'contact_forces'):
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

@configclass
class HumanoidFullModularEnvCfg_PLAY(HumanoidFullModularEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 3
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


"""
TODO:

1) rotor_inertia (=joint armature) is not set yet (v)
2) In env.scene["robot"].data, body_names and joint_names are not in the correct order.
3) Camera View (v)
4) Velocity Arrow change / World-frame vel command (v)
5) Foot step arrow visualization (v)
6) Plot contact forces (v)
7) Termination reward (v)
8) Training speed too slow
9) URDF color update (v)
10) Video recording speed to real-time (v)
11) Joint Jacobian transfer
12) Keyboard control of the velocity command (v)
13) Option to disable logging (v)
14) soft joint torque / vel limit is weird? (v)
15) Rest of the config setting match with IsaacGym (v)
16) Video recording / Screenshot / Animation for play script (v)
17) Make full joint urdf / arm only urdf
18) Find the difference between lab vanilla and IsaacGyn vanilla #! Because of normalization?
19) Rendering speed with --cpu is too slow
20) Save the code / Load the cfg files when running play script
21) Non-noisy critic observation (v)
22) With camera, I cannot start the training? (v)

"""


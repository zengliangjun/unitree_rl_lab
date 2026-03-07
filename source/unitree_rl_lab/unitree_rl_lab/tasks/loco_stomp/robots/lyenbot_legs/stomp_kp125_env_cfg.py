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

from isaaclab.envs import mdp


from unitree_rl_lab.assets.robots.lyenbot_legs_kp125 import LYENBOT_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.loco_stomp.mdp import commands, rewards, curriculums, events, observations
from isaaclab.envs.mdp import UniformVelocityCommandCfg


COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    add_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.6, 1.5),
            "operation": "scale",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-0.5, 0.5),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.5, 1.5),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale"
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.6, 1.4),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # interval
    push_robot = EventTerm(
        func=events.push_by_setting_velocity_with_level,
        mode="interval",
        interval_range_s=(3.0, 8.0),
        params={
            "velocity_range": {"x": (-0.051, 0.051), "y": (-0.051, 0.051)},
            "max_velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            "speed": 1.05},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    stomp_command = commands.StompCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),

        rel_standing_envs=0.2,
        debug_vis=True,

        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.05, 0.05), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.25, 0.25), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),

        period = 1.6,
        offset=(0.0, 0.5),
        threshold=0.55
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True #, clip={".*": (-5.0, 5.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.15, n_max=0.15))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "stomp_command"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            self.history_length = 10
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "stomp_command"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # phase = ObsTerm(func=observations.gait_phase, params={"command_name": "stomp_command"})

        def __post_init__(self):
            self.history_length = 10

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    reward_zero_lin_vel_xy = RewTerm(
        func=rewards.reward_zero_lin_vel_xy_exp,
        weight=0.45,
        params={"std": math.sqrt(0.09)},
    )

    reward_zero_ang_vel_z = RewTerm(
        func=rewards.reward_zero_ang_vel_z_exp,
        weight=0.45, params={"std": math.sqrt(0.09)}
    )
    reward_track_pitch = RewTerm(
        func=rewards.reward_track_pitch,
        weight=5.5,
        params={
            "command_name": "stomp_command",
            "std": 0.05,

            "max_stomp": -0.45,
            "target_stomp": -0.45, # -0.1,
            "speed": 1.01,

            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_hip_pitch_joint",
                    "right_hip_pitch_joint"])},
    )
    penalize_track_pitch = RewTerm(
        func=rewards.penalize_track_pitch,
        weight= -0.25,
        params={
            "command_name": "stomp_command",
            "std": 0.05,

            "max_stomp": -0.45,
            "target_stomp": -0.45, # -0.1,
            "speed": 1.01,

            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_hip_pitch_joint",
                    "right_hip_pitch_joint"])},
    )

    reward_feet_clearance = RewTerm(
        func=rewards.reward_foot_clearance,
        weight=2.7,
        params={
            "command_name": "stomp_command",

            "std": 0.02,

            "max_height": 0.13,
            "target_height": 0.13, # 0.03,
            "speed": 1.01,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_contact_point", "right_foot_contact_point"],
                preserve_order=True),
        },
    )

    penalize_feet_clearance = RewTerm(
        func=rewards.penalize_foot_clearance,
        weight= -0.17,
        params={
            "command_name": "stomp_command",

            "std": 0.02,

            "max_height": 0.13,
            "target_height": 0.13, # 0.03,
            "speed": 1.01,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_contact_point", "right_foot_contact_point"],
                preserve_order=True),
        },
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    com_support = RewTerm(
        func=rewards.com_support,
        weight=2.5,
        params={
                "command_name": "stomp_command",
                "std": 0.06,
                "target_width": 0.232,
                "asset_cfg":
                SceneEntityCfg("robot",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link"
                        ],
                    preserve_order=True)},
    )

    reward_constraint_width = RewTerm(
        func=rewards.track_constraint_width,
        weight=0.26,
        params={
            "asset_cfg":
            SceneEntityCfg("robot", body_names=[
                "left_knee_pitch_link",
                "right_knee_pitch_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link"],
            preserve_order=True),
            "target_width": 0.232,
            "std": 0.06
        },
    )

    # -- base
    penalize_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    penalize_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)

    penalize_ankle_joint_vel = RewTerm(func=mdp.joint_vel_l2,
                          weight=-2e-3,
                          params={"asset_cfg":
                                  SceneEntityCfg("robot", joint_names=".*ankle_.*")})

    penalize_yaw_joint_vel = RewTerm(func=mdp.joint_vel_l2,
                          weight=-1e-3,
                          params={"asset_cfg":
                                  SceneEntityCfg("robot", joint_names=[".*hip_yaw.*", ".*waist_yaw.*"])})

    penalize_pitch_joint_vel = RewTerm(func=mdp.joint_vel_l2,
                          weight=-6e-4,
                          params={"asset_cfg":
                                  SceneEntityCfg("robot", joint_names=[".*hip_pitch.*",
                                                                       ".*hip_roll.*",
                                                                       ".*knee_pitch.*"])})

    penalize_ankle_joint_acc = RewTerm(func=mdp.joint_acc_l2,
                          weight=-7.5e-8,
                          params={"asset_cfg":
                                  SceneEntityCfg("robot", joint_names=".*ankle_.*")})

    penalize_yaw_joint_acc = RewTerm(func=mdp.joint_acc_l2,
                          weight=-7.5e-7,
                          params={"asset_cfg":
                                  SceneEntityCfg("robot", joint_names=[".*hip_yaw.*", ".*waist_yaw.*"])})

    penalize_pitch_joint_acc = RewTerm(func=mdp.joint_acc_l2,
                          weight=-3.5e-8,
                          params={"asset_cfg":
                                  SceneEntityCfg("robot", joint_names=[".*hip_pitch.*",
                                                                       ".*hip_roll.*",
                                                                       ".*knee_pitch.*"])})


    penalize_ankle_action_rate = RewTerm(func=rewards.action_rate_l2_ext,
                          weight=-3.5e-3,
                          params={"asset_cfg":
                                  SceneEntityCfg("robot", joint_names=".*ankle_.*")})

    penalize_yaw_action_rate = RewTerm(func=rewards.action_rate_l2_ext,
                          weight=-6e-2,
                          params={"asset_cfg":
                                  SceneEntityCfg("robot",
                                                 joint_names=[".*hip_yaw.*", ".*waist_yaw.*"])})

    penalize_pitch_action_rate = RewTerm(func=rewards.action_rate_l2_ext,
                          weight=-3.5e-3,
                          params={"asset_cfg":
                                  SceneEntityCfg("robot",
                                                 joint_names=[".*hip_pitch.*",
                                                              ".*hip_roll.*",
                                                              ".*knee_pitch.*"])})

    penalize_dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-8.0)

    penalize_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                    ".*_hip_yaw_joint"
                ],
            )
        },
    )
    penalize_joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                    ".*_hip_roll.*",
                    ".*_ankle_pitch.*",
                    ".*_ankle_roll.*"])},
    )

    penalize_stand_deviation = RewTerm(
        func=rewards.stand_deviation_l1,
        weight=-0.45,
        params={"command_name": "stomp_command",
                "asset_cfg": SceneEntityCfg("robot")},
    )

    penalize_energy = RewTerm(func=rewards.energy, weight=-2e-5)

    # -- robot
    penalize_flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    penalize_base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.82})

    # -- feet
    reward_gait = RewTerm(
        func=rewards.track_feet_gait,
        weight=1.5,
        params={
            "command_name": "stomp_command",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True),
        },
    )
    penalize_gait = RewTerm(
        func=rewards.penalize_feet_gait,
        weight= -0.9,
        params={
            "command_name": "stomp_command",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True),
        },
    )
    # reward_euler = RewTerm(
    #     func=rewards.reward_euler,
    #     weight=0.35,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
    #             preserve_order=True),
    #     },
    # )
    penalty_feet_orientation = RewTerm(
        func=rewards.penalty_orientation,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True),
        },
    )
    penalize_feet_slide = RewTerm(
        func=rewards.feet_slide,
        weight=-0.35,
        params={
            "command_name": "stomp_command",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")
        },
    )

    # -- other
    penalize_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-330.0,
    )

    def __post_init__(self):
        pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    push_levels = CurrTerm(func=curriculums.push_levels,
        params={
                "command_term_name": "stomp_command",
                "event_term_name": "push_robot",
                "reward_term_name": "reward_feet_clearance",
            })

@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 2
        self.curriculum = None

        self.commands.stomp_command.ranges=self.commands.stomp_command.limit_ranges

        self.rewards.reward_track_pitch.params["target_stomp"] = self.rewards.reward_track_pitch.params["max_stomp"]
        self.rewards.reward_feet_clearance.params["target_height"] = self.rewards.reward_feet_clearance.params["max_height"]

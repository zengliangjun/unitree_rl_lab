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

from isaaclab.envs.mdp import observations, events, terminations
from isaaclab.envs.mdp.actions import actions_cfg

from unitree_rl_lab.tasks.locomotion.mdp import curriculums as rl_curriculums
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import curriculums
from isaaclab_tasks.manager_based.classic.humanoid.mdp import rewards as humanoid_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import rewards
from isaaclab.envs.mdp import rewards as mdp_rewards

from unitree_rl_lab.tasks.locomotionv2.mdp import observations as mdp_observations
from unitree_rl_lab.tasks.locomotionv2.mdp import rewards_ext

from unitree_rl_lab.assets.robots.lyenbot_legs import LYENBOT_CFG as ROBOT_CFG

from unitree_rl_lab.tasks.locomotionv2.mdp import commands


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
        func=events.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    base_mass = EventTerm(
        func=events.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    robot_joint_stiffness_and_damping = EventTerm(
        func=events.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_pos_limits = EventTerm(
        func=events.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.15, 1.5),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    mass = EventTerm(
        func=events.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.6, 1.5),
            "operation": "scale",
        },
    )

    body_com = EventTerm(
        func=events.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_range": {
                "x": (-0.06, 0.06),
                "y": (-0.06, 0.06),
                "z": (-0.06, 0.06),
            },
        },
    )

    base_external_force_torque = EventTerm(
        func=events.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-0.5, 0.5),
        },
    )
    reset_base = EventTerm(
        func=events.reset_root_state_uniform,
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
        func=events.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=events.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = commands.CommandWithPhaseCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),

        rel_standing_envs=0.2,

        rel_heading_envs=1.0,
        heading_command=False,

        debug_vis=False,
        ranges=commands.CommandWithPhaseCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=commands.CommandWithPhaseCfg.Ranges(
            lin_vel_x=(-1, 1), lin_vel_y=(-1, 1), ang_vel_z=(-1, 1)
        ),

        period=0.6,  #0.8,
        offset=(0.0, 0.5),
        threshold=0.55
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = actions_cfg.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True #, clip={".*": (-5.0, 5.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=observations.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=observations.projected_gravity, noise=Unoise(n_min=-0.15, n_max=0.15))
        velocity_commands = ObsTerm(func=observations.generated_commands, params={"command_name": "base_velocity"})
        stomp_commands = ObsTerm(func=mdp_observations.stomp_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=observations.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_rel = ObsTerm(func=observations.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=observations.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):

        base_lin_vel = ObsTerm(func=observations.base_lin_vel)
        base_ang_vel = ObsTerm(func=observations.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=observations.projected_gravity)
        velocity_commands = ObsTerm(func=observations.generated_commands, params={"command_name": "base_velocity"})
        stomp_commands = ObsTerm(func=mdp_observations.stomp_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=observations.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=observations.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=observations.last_action)

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=rewards.track_lin_vel_xy_yaw_frame_exp,
        weight=1.6,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=rewards.track_ang_vel_z_world_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp_rewards.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp_rewards.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp_rewards.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp_rewards.joint_vel_l2, weight=-0.003) # weight=-0.001)
    joint_acc = RewTerm(func=mdp_rewards.joint_acc_l2, weight=-7.5e-7) # weight=-2.5e-7)
    action_rate = RewTerm(func=mdp_rewards.action_rate_l2, weight=-0.05)
    # ankle_action_rate = RewTerm(func=rewards_ext.action_rate_l2_ext,
    #                       weight=-0.03,
    #                       params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*ankle_.*")})

    dof_pos_limits = RewTerm(func=mdp_rewards.joint_pos_limits, weight=-5.0)

    energy = RewTerm(
        func=humanoid_rewards.power_consumption,
        weight=-2e-5,
        params={
            "gear_ratio": {
                ".*_hip_pitch_.*": 5.0,
                ".*_hip_roll_.*": 15.0,
                ".*_hip_yaw_.*": 50.0,
                ".*_knee_pitch_.*": 5.0,
                ".*_ankle_pitch_.*": 22.0,
                ".*_ankle_roll_.*": 22.0,
                "waist_yaw_.*": 100.0,
            }
        },
    )

    joint_deviation = RewTerm(
        func=mdp_rewards.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_yaw_.*",
                    ".*_hip_yaw_.*",
                    ".*_ankle_roll_.*"
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp_rewards.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                    ".*_hip_roll.*",
                    ".*_ankle_pitch.*"])},
    )

    stand_deviation = RewTerm(
        func=rewards_ext.stand_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp_rewards.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp_rewards.base_height_l2, weight=-10, params={"target_height": 0.82})

    # -- feet
    reward_gait = RewTerm(
        func=rewards_ext.track_feet_gait,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True),
        },
    )
    penalize_gait = RewTerm(
        func=rewards_ext.penalize_feet_gait,
        weight= -0.2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True),
        },
    )
    penalty_feet_orientation = RewTerm(
        func=rewards_ext.penalty_orientation,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True),
        },
    )

    feet_slide = RewTerm(
        func=rewards_ext.feet_slide,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")
        },
    )

    feet_width = RewTerm(
        func=rewards_ext.reward_feet_width,
        weight=0.3,
        params={
            "asset_cfg":
            SceneEntityCfg("robot", body_names=[
                # "left_knee_pitch_link",
                # "right_knee_pitch_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link"],
            preserve_order=True),
            "target_width": 0.275, # 0.32, # 0.23
            "std": 0.12
        },
    )

    reward_feet_clearance = RewTerm(
        func=rewards_ext.reward_foot_clearance,
        weight= 0.24,
        params={
            "command_name": "base_velocity",

            "std": 0.02,

            "max_height": 0.05,
            "target_height": 0.05, # 0.03,
            "speed": 1.01,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_contact_point", "right_foot_contact_point"],
                preserve_order=True),
        },
    )

    penalize_feet_clearance = RewTerm(
        func=rewards_ext.penalize_foot_clearance,
        weight= -0.095,
        params={
            "command_name": "base_velocity",

            "std": 0.02,

            "max_height": 0.05,
            "target_height": 0.05, # 0.03,
            "speed": 1.01,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_contact_point", "right_foot_contact_point"],
                preserve_order=True),
        },
    )

    penalize_feet_forces = RewTerm(
        func=rewards_ext.penalize_feet_forces,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "threshold": 700,
            "max_over_penalize_forces": 400,
            },
    )
    # -- other
    undesired_contacts = RewTerm(
        func=mdp_rewards.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    def __post_init__(self):
        pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=terminations.time_out, time_out=True)
    base_height = DoneTerm(func=terminations.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=terminations.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=curriculums.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(func=rl_curriculums.lin_vel_cmd_levels)



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
        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 4
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.curriculum = None

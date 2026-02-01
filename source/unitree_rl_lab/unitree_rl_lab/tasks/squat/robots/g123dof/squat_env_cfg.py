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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG as ROBOT_CFG

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.squat.mdp import command_squat_cfg, events, observations, rewards, curriculums

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

    #
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    def __post_init__(self):
        # self.robot.spawn.articulation_props.enabled_self_collisions = False
        pass


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
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.5, 3.0),
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
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
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
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.25, 0.25),
            "velocity_range": (-0.25, 0.25),
        },
    )

    # interval
    push_robot = EventTerm(
        func=events.push_by_setting_velocity_with_level,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={
            "velocity_range": {"x": (-0.051, 0.051), "y": (-0.051, 0.051)},
            "max_velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            "speed": 1.1},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    squat_command = command_squat_cfg.SquatCommandCfg(
        asset_cfg = SceneEntityCfg( "robot",
                joint_names=[
                    ".*_knee_.*",
                ],
                preserve_order=True),

        resampling_time_range=(4, 13),
        rel_reset_init_envs=0.3,
        rel_compute_init_envs=0.3,
        rel_compute_max_envs=0.3,

        ranges=command_squat_cfg.SquatCommandCfg.Ranges(
            squat_phase = (- math.pi * 15 / 32, 0.8599),
            full_times = (4, 6.5)
        ),

        max_limit_ranges=command_squat_cfg.SquatCommandCfg.Ranges(
            squat_phase = ( - math.pi * 15 / 32, 0.8599),
            full_times = (1.2, 6.5)
        ),

        min_limit_ranges=command_squat_cfg.SquatCommandCfg.Ranges(
            squat_phase = ( - math.pi * 15 / 32, 0.8599),
            full_times = (4, 4.5)
        )

    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        squat_commands = ObsTerm(func=observations.squat_command, params={"command_name": "squat_command"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
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
        squat_commands = ObsTerm(func=observations.squat_command, params={"command_name": "squat_command"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_squat_pos = RewTerm(
        func=rewards.track_squat_pos_exp,
        weight=2.5,
        params={"command_name": "squat_command",
                "std": math.sqrt(0.01),
                "asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_knee_joint",
                        "right_knee_joint"],
                    preserve_order=True)},
    )
    penalty_squat_pos = RewTerm(
        func=rewards.track_squat_error,
        weight=- 1e-3,
        params={"command_name": "squat_command",
                "finished_weight": 1,
                "finished_max_weight": 2,
                "penalty_weight": 1,
                "penalty_max_weight": 3,
                "std": math.sqrt(0.01),
                "asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_knee_joint",
                        "right_knee_joint"],
                    preserve_order=True)}
    )
    track_symmetry_pos = RewTerm(
        func=rewards.track_symmetry_pos_exp,
        weight=0.15,
        params={"std": math.sqrt(0.09),
                "asset_cfg":
                SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "left_ankle_pitch_joint",
                        "right_ankle_pitch_joint",
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "left_elbow_joint",
                        "right_elbow_joint"],
                    preserve_order=True)},
    )

    reward_pitch2zero = RewTerm(
        func=rewards.reward_pitch2zero,
        weight=0.25,
        params={"std": 0.36,
                "command_name": "squat_command",
                "asset_cfg":
                SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "left_knee_joint",
                        "right_knee_joint",
                        "left_ankle_pitch_joint",
                        "right_ankle_pitch_joint"
                        ],
                    preserve_order=True)},
    )

    # only for static is com is zero
    com_zero = RewTerm(
        func=rewards.com_zero,
        weight=5,
        params={"std": 0.12,
                "command_name": "squat_command",
                "asset_cfg":
                SceneEntityCfg("robot",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link"
                        ],
                    preserve_order=True)},
    )
    zero_ang_vel = RewTerm(
        func=rewards.reward_zero_ang_vel_exp_v1,
        weight=0.8,
        params={"finished_weight": 3, "std": 0.25}
    )
    zero_lin_xy_vel = RewTerm(
        func=rewards.reward_zero_lin_vel_xy_exp_v1,
        weight=0.4,
        params={"finished_weight": 3, "std": 0.25}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.13)

    joint_vel = RewTerm(func=rewards.joint_vel_l2, weight=-0.01,
        params={"finished_weight": 3} )
    joint_acc = RewTerm(func=rewards.joint_acc_l2, weight=-1e-5,
        params={"finished_weight": 3} )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=rewards.energy, weight=-4e-3,
        params={"finished_weight": 3})

    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll_.*",
                    ".*_shoulder_yaw_.*",
                    ".*_elbow_.*",
                    ".*_wrist_roll_.*",
                    "waist.*",
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_ankle_roll_joint"
                ],
            )
        },
    )

    track_constraint_width = RewTerm(
        func=rewards.track_constraint_width,
        weight=0.1,
        params={
            "asset_cfg":
            SceneEntityCfg("robot", body_names=[
                "left_knee_link",
                "right_knee_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link"],
            preserve_order=True),
            "target_width": 0.232,
            "std": 0.12
        },
    )

    sample_force = RewTerm(
        func=rewards.contact_same_force,
        weight=0.2,
        params={
            "sensor_cfg":
            SceneEntityCfg("contact_forces", body_names=[
                "left_ankle_roll_link",
                "right_ankle_roll_link"],
            preserve_order=True)
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces",
            body_names=[
            'pelvis',
            'left_hip_pitch_link',
            'left_hip_roll_link',
            'left_hip_yaw_link',
            'left_knee_link',
            'right_hip_pitch_link',
            'right_hip_roll_link',
            'right_hip_yaw_link',
            'right_knee_link',
            'torso_link',
            'left_shoulder_pitch_link',
            'left_shoulder_roll_link',
            'left_shoulder_yaw_link',
            'left_elbow_link',
            'left_wrist_roll_rubber_hand',
            'right_shoulder_pitch_link',
            'right_shoulder_roll_link',
            'right_shoulder_yaw_link',
            'right_elbow_link',
            'right_wrist_roll_rubber_hand'],
            preserve_order = True),
        },
    )
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-330.0,
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.15})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    squat_levels = CurrTerm(func=curriculums.squat_cmd_levels,
        params={
                "command_term_name": "squat_command",
                "reward_term_name": "track_squat_pos",
            })

    push_levels = CurrTerm(func=curriculums.squat_push_levels,
        params={
                "command_term_name": "squat_command",
                "event_term_name": "push_robot",
                "reward_term_name": "track_squat_pos",
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
        self.episode_length_s = 26
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if hasattr(self.scene, "contact_forces") and self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if hasattr(self.scene, "height_scanner") and self.scene.height_scanner is not None:
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
        self.episode_length_s = 200.0
        self.scene.num_envs = 32

        self.scene.terrain.terrain_generator.border_width=2.0
        self.scene.terrain.terrain_generator.num_rows=1
        self.scene.terrain.terrain_generator.num_cols=1
        self.commands.squat_command.ranges = self.commands.squat_command.max_limit_ranges
        self.events.push_robot = None
        self.curriculum = None

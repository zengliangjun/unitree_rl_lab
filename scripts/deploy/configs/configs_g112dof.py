from dataclasses import MISSING
try:
    from isaaclab.utils import configclass
except:
    from configclasses import configclass

from . import configs


@configclass
class G112dofConfig(configs.MujocoConfig):

    sim_config = configs.IsaacSimConfig(
        joint_names= ["left_hip_pitch_joint",
                        "left_hip_roll_joint",
                        "left_hip_yaw_joint",
                        "left_knee_joint",
                        "left_ankle_pitch_joint",
                        "left_ankle_roll_joint",
                        "right_hip_pitch_joint",
                        "right_hip_roll_joint",
                        "right_hip_yaw_joint",
                        "right_knee_joint",
                        "right_ankle_pitch_joint",
                        "right_ankle_roll_joint"
                    ],

        actuator_stiffness = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40],

        actuator_damping = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2],

        init_pos = [-0.1,  0.0,  0.0,  0.3, -0.2, 0.0,
                  -0.1,  0.0,  0.0,  0.3, -0.2, 0.0],

        ang_vel_objsscale = [0.25, 0.25, 0.25],
        gravity_objsscale = [1, 1, 1],
        commands_objsscale = [2.0, 2.0, 0.25],
        joint_pos_objsscale = [1, 1, 1,
                               1, 1, 1,
                               1, 1, 1,
                               1, 1, 1],
        joint_vel_objsscale = [0.05, 0.05, 0.05,
                               0.05, 0.05, 0.05,
                               0.05, 0.05, 0.05,
                               0.05, 0.05, 0.05],
        action_objsscale = [1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1],

        observations_names = ["ang_vel", "gravity", "commands", "joint_pos", "joint_vel", "action", "phase"],
        observations_length = 1,

        simulation_dt = 0.002,
        decimation = 10,
        action_scale = 0.25,
    )

    policy_path = "/workspace/HUMANOID/unitree_rl_gym/deploy/pre_train/g1/motion.pt"
    mujoco_path = "/workspace/HUMANOID/unitree_rl_gym/resources/robots/g1_description/scene.xml"

    simulation_duration = 20

    command = [0.5, 0, 0]

    def __post_init__(self):
        self.sim_config.phase_objsscale = [1, 1]


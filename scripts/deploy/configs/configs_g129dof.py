from dataclasses import MISSING
try:
    from isaaclab.utils import configclass
except:
    from configclasses import configclass

from . import configs


@configclass
class G129dofConfig(configs.MujocoConfig):

    sim_config = configs.IsaacSimConfig(
        joint_names= ["left_hip_pitch_joint",
                            "right_hip_pitch_joint",
                            "waist_yaw_joint",
                            "left_hip_roll_joint",
                            "right_hip_roll_joint",
                            "waist_roll_joint",
                            "left_hip_yaw_joint",
                            "right_hip_yaw_joint",
                            "waist_pitch_joint",
                            "left_knee_joint",
                            "right_knee_joint",
                            "left_shoulder_pitch_joint",
                            "right_shoulder_pitch_joint",
                            "left_ankle_pitch_joint",
                            "right_ankle_pitch_joint",
                            "left_shoulder_roll_joint",
                            "right_shoulder_roll_joint",
                            "left_ankle_roll_joint",
                            "right_ankle_roll_joint",
                            "left_shoulder_yaw_joint",
                            "right_shoulder_yaw_joint",
                            "left_elbow_joint",
                            "right_elbow_joint",
                            "left_wrist_roll_joint",
                            "right_wrist_roll_joint",
                            "left_wrist_pitch_joint",
                            "right_wrist_pitch_joint",
                            "left_wrist_yaw_joint",
                            "right_wrist_yaw_joint"],

        actuator_stiffness = [100., 100., 200., 100., 100.,  40., 100., 100.,  40., 150., 150.,  40.,
          40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,
          40.,  40.,  40.,  40.,  40.],

        actuator_damping = [2., 2., 5., 2., 2., 5., 2., 2., 5., 4., 4., 1., 1., 2., 2., 1., 1., 2.,
         2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],

        init_pos = [-0.1000, -0.1000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.3000,  0.3000,  0.3000,  0.3000, -0.2000, -0.2000,  0.2500,
         -0.2500,  0.0000,  0.0000,  0.0000,  0.0000,  0.9700,  0.9700,  0.1500,
         -0.1500,  0.0000,  0.0000,  0.0000,  0.0000],

        ang_vel_objsscale = [0.2, 0.2, 0.2],
        gravity_objsscale = [1, 1, 1],
        commands_objsscale = [1, 1, 1],
        joint_pos_objsscale = [1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1],
        joint_vel_objsscale = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                           0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                           0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                           0.05, 0.05],
        action_objsscale = [1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1],

        observations_names = ["ang_vel", "gravity", "commands", "joint_pos", "joint_vel", "action"],
        observations_length = 5,

        simulation_dt = 0.005,
        decimation = 4,
        action_scale = 0.25,
    )

    policy_path = "/workspace/data2/VSCODE/RL/ISAACSIM45LAB2/unitree_rl_lab/logs/rsl_rl/unitreesteps/2025-09-18_18-26-06/exported/policy.pt"
    mujoco_path = "/workspace/VS2025/SIMULATION/ISAACSIM45ENVS/unitree_ros/robots/g1_description/g1_29dof.xml"

    simulation_duration = 20

    command = [0.5, 0, 0]


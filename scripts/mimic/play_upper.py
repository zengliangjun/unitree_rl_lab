"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python csv_to_npz.py -f path_to_input.csv --input_fps 60
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", "-f", type=str, required=False, help="The path to the input motion csv file.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.input_file = "/workspace/PROJECTS/MOTIONS/data/AMASS/g1_23dof_50fps_pos_only/BMLmovi/Subject_8_F_2_stageii.pkl"
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os
import os.path as osp
import sys
work_root = osp.join(osp.dirname(__file__), "../..")
os.chdir(work_root)

source_root = osp.join(work_root, "source/unitree_rl_lab")
if source_root not in sys.path:
    sys.path.insert(0, source_root)


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
##
# Pre-defined configs
##
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG as ROBOT_CFG  # Currently only support G1-29dof
import pickle

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def __post_init__(self):
        super(ReplayMotionsSceneCfg, self).__post_init__()
        self.robot.spawn.articulation_props.fix_root_link = True


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        device: torch.device,
    ):
        self.motion_file = motion_file
        self.current_idx = 0
        self.device = device
        self._load_motion()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        '''
        if self.motion_file.endswith("txt"):
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=","
                )
            )
        el
        '''

        if self.motion_file.endswith("pkl"):
            with open(self.motion_file, "rb") as fd:
                motion = pickle.load(fd)

        self.dof_pos = motion["dof_pos"]
        self.output_frames = self.dof_pos.shape[0]

    def get_next_state(
        self,
    ) -> tuple[torch.Tensor, bool]:
        """Gets the next state of the motion."""
        dof_pos = self.dof_pos[self.current_idx : self.current_idx + 1]
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        dof_pos = torch.tensor(dof_pos, dtype=torch.float32, device = self.device)

        return dof_pos, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        device=sim.device,
    )

    # Extract scene entities
    robot = scene["robot"]

    joint_names = []
    for name in  scene.cfg.robot.joint_sdk_names:
        if 0 == len(name):
            continue
        joint_names.append(name)

    upper_names = ["waist_yaw_joint",
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint"]

    sdk_upper_index = []
    for name in upper_names:
        sdk_upper_index.append(joint_names.index(name))


    sim_upper_indexes = robot.find_joints(upper_names, preserve_order=True)[0]

    # Simulation loop
    while simulation_app.is_running():
        (
            dof_pos,
            reset_flag,
        ) = motion.get_next_state()

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, sim_upper_indexes] = dof_pos[:, sdk_upper_index]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / 50
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

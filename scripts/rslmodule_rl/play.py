# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

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

import gymnasium as gym
import os
import time
import torch


import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

from isaaclabext_rl.rsl_rl.rl_cfg import RslRlModularOnPolicyRunnerCfg
from isaaclabext_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
from rslext_rl.runners.modular_on_policy_runner import ModularOnPolicyRunner

from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlModularOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model

    runner = ModularOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    runner.load(resume_path)

    # obtain the trained policy for inference
    leg_policy, arm_policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        leg_policy_nn = runner.leg_alg.policy
        arm_policy_nn = runner.arm_alg.policy
    except AttributeError:
        # version 2.2 and below
        leg_policy_nn = runner.leg_alg.actor_critic
        arm_policy_nn = runner.arm_alg.actor_critic

    # extract the normalizer
    if hasattr(leg_policy_nn, "actor_obs_normalizer"):
        leg_normalizer = leg_policy_nn.actor_obs_normalizer
        arm_normalizer = arm_policy_nn.actor_obs_normalizer
    elif hasattr(leg_policy_nn, "student_obs_normalizer"):
        leg_normalizer = leg_policy_nn.student_obs_normalizer
        arm_normalizer = arm_policy_nn.student_obs_normalizer
    else:
        leg_normalizer = None
        arm_normalizer = None
    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(leg_policy_nn, normalizer=leg_normalizer, path=export_model_dir, filename="leg_policy.pt")
    export_policy_as_jit(arm_policy_nn, normalizer=arm_normalizer, path=export_model_dir, filename="arm_policy.pt")
    export_policy_as_onnx(leg_policy_nn, normalizer=leg_normalizer, path=export_model_dir, filename="leg_policy.onnx")
    export_policy_as_onnx(arm_policy_nn, normalizer=arm_normalizer, path=export_model_dir, filename="arm_policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if isinstance(obs, list) or isinstance(obs, tuple):
        obs = obs[0]
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            leg_actions = leg_policy(obs)
            arm_actions = arm_policy(obs)
            actions = torch.cat([leg_actions, arm_actions], dim=-1)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

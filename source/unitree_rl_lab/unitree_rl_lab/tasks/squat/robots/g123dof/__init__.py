import gymnasium as gym

gym.register(
    id="g123dof-cam-squat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camenv_cfg:G123EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.camenv_cfg:G123PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.squat.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="g123dofcamsquatv1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camenv_cfg:G123EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.camenv_cfg:G123PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.squat.agents.rsl_rl_ppo_cfg:BasePPORunnerCfgV1",
    },
)

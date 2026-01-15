import gymnasium as gym

gym.register(
    id="lyenbot-leg-squat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.squat_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.squat.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="lyenbotleg-unitree-squat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_env_cfg:UnitreeRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.squat_env_cfg:UnitreeRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.squat.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

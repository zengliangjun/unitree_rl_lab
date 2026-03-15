import gymnasium as gym

gym.register(
    id="lyenbotlegs-kp125-stomp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stomp_kp125_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stomp_kp125_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.loco_stomp.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="lyenbotlegs-kp160-stomp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stomp_kp160_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stomp_kp160_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.loco_stomp.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="lyenbotlegs-kp100-stomp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stomp_kp100_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stomp_kp100_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

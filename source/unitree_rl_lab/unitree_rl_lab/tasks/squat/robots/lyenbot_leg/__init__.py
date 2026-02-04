import gymnasium as gym

gym.register(
    id="lyenbotlegcamsquatv1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_camenv_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.squat_camenv_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.squat.agents.rsl_rl_ppo_cfg:BasePPORunnerCfgV1",
    },
)

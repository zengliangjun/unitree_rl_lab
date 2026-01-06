import gymnasium as gym

gym.register(
    id="Unitree-G1-23dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

'''
gym.register(
    id="Modular-G1-23dof-Velocity",
    entry_point="isaaclab_ext.envs.manager_based_rl_env:ModuleRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.modular_env_cfg:HumanoidFullModularEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.modular_env_cfg:HumanoidFullModularEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rslrl_modular_cfg:ModularPPORunnerCfg",
    },
)
'''


gym.register(
    id="Modular-G1-23dof-Velocity-v2",
    entry_point="isaaclab_ext.envs.manager_based_rl_env:ModuleRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.modular_env_cfg_v2:HumanoidFullModularEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.modular_env_cfg_v2:HumanoidFullModularEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rslrl_modular_cfg:ModularPPORunnerCfg",
    },
)


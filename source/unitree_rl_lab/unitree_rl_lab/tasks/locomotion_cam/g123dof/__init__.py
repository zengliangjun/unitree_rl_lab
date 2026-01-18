import gymnasium as gym

gym.register(
    id="g1-23dof-cam",
    entry_point="isaaclab_ext.envs.manager_based_rl_env:ModuleRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.modular_env_cfg_fix:EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.modular_env_cfg_fix:EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion_cam.agents.rslrl_modular_cfg:ModularPPORunnerCfg",
    },
)

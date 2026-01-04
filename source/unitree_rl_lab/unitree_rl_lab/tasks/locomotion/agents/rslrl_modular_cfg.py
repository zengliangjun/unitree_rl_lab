# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlOnPolicyRunnerCfg
)

from isaaclabext_rl.rsl_rl.rl_cfg import RslRlModularOnPolicyRunnerCfg

@configclass
class ModularPPORunnerCfg(RslRlModularOnPolicyRunnerCfg):
    seed = 42 # -1
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 200
    experiment_name = ""
    enable_logging = True
    wandb_project = ""
    store_code_state = True
    leg_policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu"
    )
    arm_policy = RslRlPpoActorCriticCfg(
        init_noise_std=1e-1,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu"
    )
    leg_algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.12143, # 0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.00041, # 1.e-5,
        schedule="adaptive",
        gamma=0.9751, # 0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.,
    )
    arm_algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.,
    )

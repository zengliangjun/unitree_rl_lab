from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from unitree_rl_lab.tasks.locomotion.mdp import commands

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class action_rate_l2_withname(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        action_name = cfg.params.get("action_name", None)
        assert action_name is not None, "action_name must be specified in the reward term config."

        start_dim = 0
        end_dim = 0
        for name, dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
            start_dim = end_dim
            end_dim += dim
            if name != action_name:
                continue

        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, env: ManagerBasedRLEnv,
                action_name: str) -> torch.Tensor:

        error = torch.square(env.action_manager.action - env.action_manager.prev_action)[:, self.start_dim: self.end_dim]
        return torch.sum(error, dim=1)

class action_rate_l2_withname_ext(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        action_name = cfg.params.get("action_name", None)
        assert action_name is not None, "action_name must be specified in the reward term config."

        start_dim = 0
        end_dim = 0
        for name, dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
            start_dim = end_dim
            end_dim += dim
            if name != action_name:
                continue

        self.start_dim = start_dim
        self.end_dim = end_dim

        self.prev_prev_action = torch.zeros(env.num_envs, self.end_dim - self.start_dim, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.prev_prev_action.zero_()
        else:
            self.prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv,
                action_name: str) -> torch.Tensor:

        error = (env.action_manager.action - 2 * env.action_manager.prev_action)[:, self.start_dim: self.end_dim] + self.prev_prev_action
        error = torch.square(error)

        self.prev_prev_action[...] = env.action_manager.prev_action[:, self.start_dim: self.end_dim]
        return torch.sum(error, dim=1)

def is_alive_exp(env: ManagerBasedRLEnv, command_name: str, std: float = 0.05) -> torch.Tensor:
    """Reward for being alive."""
    command: commands.CommandWithEpisodeInfo = env.command_manager.get_term(command_name)
    aliving = (command.average_episode_length / env.max_episode_length) / std

    return (~env.termination_manager.terminated).float() * math.exp(- aliving)



class reward_feet_air_time(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.last_contacts = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self.feet_air_time = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if len(env_ids) == 0:
            return
        self.last_contacts[env_ids] = 0
        self.feet_air_time[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        contacts_threshold: float,
        period: float = 0.8,
        offset: list[float] = [0.0, 0.5],
        stance_threshold: float = 0.55,
        command_name="base_velocity"):

        # Compute stance mask
        global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
        phases = []
        for offset_ in offset:
            phase = (global_phase + offset_) % 1.0
            phases.append(phase)
        leg_phase = torch.cat(phases, dim=-1)

        stance_mask = leg_phase < stance_threshold

        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
        contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > contacts_threshold

        contact_filt = torch.logical_or(torch.logical_or(contacts, stance_mask), self.last_contacts)
        self.last_contacts = contacts

        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self._env.step_dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~contact_filt
        return air_time.sum(dim=1)

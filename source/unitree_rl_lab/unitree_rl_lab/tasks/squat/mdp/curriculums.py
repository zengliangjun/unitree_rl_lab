from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.managers import EventTermCfg

from . import command_squat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def squat_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_term_name: str = "suqat_command",
    reward_term_name: str = "track_squat_pos",
) -> torch.Tensor:
    command_term: command_squat.SuqatCommand = env.command_manager.get_term(command_term_name)
    ranges = command_term.cfg.ranges
    max_limit_ranges = command_term.cfg.max_limit_ranges
    min_limit_ranges = command_term.cfg.min_limit_ranges


    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    level = max_limit_ranges.full_times[0] / ranges.full_times[0]
    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.65 and command_term.mean_episode_length > env.max_episode_length * 0.6:
            #suqat_min = max(ranges.suqat_phase[0] - 0.05, max_limit_ranges.suqat_phase[0])
            #suqat_max = min(ranges.suqat_phase[1] + 0.05, max_limit_ranges.suqat_phase[1])

            #ranges.suqat_phase = [suqat_min, suqat_max]

            times_min = max(ranges.full_times[0] - 0.05, max_limit_ranges.full_times[0])
            times_max = min(ranges.full_times[1] + 0.05, max_limit_ranges.full_times[1])

            ranges.full_times = [times_min, times_max]
            if times_min <= max_limit_ranges.full_times[0] and \
                times_max >= max_limit_ranges.full_times[1] and \
                command_term.mean_episode_length > env.max_episode_length * 0.8:


                reward_term.params["finished_weight"] += 0.1
                reward_term.params["finished_weight"] = min(reward_term.params["finished_weight"], reward_term.params["finished_max_weight"])

                level += reward_term.params["finished_weight"] / reward_term.params["finished_max_weight"]

                if reward_term.params["finished_weight"] >= reward_term.params["finished_max_weight"]:

                    reward_term.params["penalty_weight"] += 0.01
                    reward_term.params["penalty_weight"] = min(reward_term.params["penalty_weight"], reward_term.params["penalty_max_weight"])

                    level += reward_term.params["penalty_weight"] / reward_term.params["penalty_max_weight"]


    else:
        times_min, times_max = ranges.full_times
        if times_min <= max_limit_ranges.full_times[0] and \
            times_max >= max_limit_ranges.full_times[1]:

            level += reward_term.params["finished_weight"] / reward_term.params["finished_max_weight"]

            if reward_term.params["finished_weight"] >= reward_term.params["finished_max_weight"]:

                level += reward_term.params["penalty_weight"] / reward_term.params["penalty_max_weight"]

        '''
        if reward < reward_term.weight * 0.2:
            suqat_min = min(ranges.suqat_phase[0] + 0.05, min_limit_ranges.suqat_phase[0])
            suqat_max = max(ranges.suqat_phase[1] - 0.05, min_limit_ranges.suqat_phase[1])

            ranges.suqat_phase = [suqat_min, suqat_max]

            times_min = min(ranges.full_times[0] + 0.05, min_limit_ranges.full_times[0])
            times_max = max(ranges.full_times[1] - 0.05, min_limit_ranges.full_times[1])

            ranges.full_times = [times_min, times_max]
        '''

    return torch.tensor(level, device=env.device)



def squat_push_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_term_name: str = "suqat_command",
    event_term_name: str = "push_robot",
    reward_term_name: str = "track_squat_pos",
) -> torch.Tensor:

    command_term: command_squat.SuqatCommand = env.command_manager.get_term(command_term_name)

    action_term: EventTermCfg = env.event_manager.get_term_cfg(event_term_name)
    ranges: dict = action_term.params["velocity_range"]
    max_range: dict = action_term.params["max_velocity_range"]
    speed: float = action_term.params["speed"]

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.75 and command_term.mean_episode_length > env.max_episode_length * 0.85:

            for key in ranges:
                org = ranges[key]
                max_org = max_range[key]

                new_min = org[0] * speed
                new_max = org[1] * speed

                new_min = max(new_min, max_org[0])
                new_max = min(new_max, max_org[1])

                ranges[key] = (new_min, new_max)

        '''
        if reward < reward_term.weight * 0.2:
            suqat_min = min(ranges.suqat_phase[0] + 0.05, min_limit_ranges.suqat_phase[0])
            suqat_max = max(ranges.suqat_phase[1] - 0.05, min_limit_ranges.suqat_phase[1])

            ranges.suqat_phase = [suqat_min, suqat_max]

            times_min = min(ranges.full_times[0] + 0.05, min_limit_ranges.full_times[0])
            times_max = max(ranges.full_times[1] - 0.05, min_limit_ranges.full_times[1])

            ranges.full_times = [times_min, times_max]
        '''

    return torch.tensor(ranges["x"][1], device=env.device)

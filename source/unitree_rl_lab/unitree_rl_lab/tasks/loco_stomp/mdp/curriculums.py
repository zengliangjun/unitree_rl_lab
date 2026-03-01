from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.managers import EventTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from unitree_rl_lab.tasks.loco_stomp.mdp import commands

def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command(command_name)
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def squat_push_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_term_name: str = "squat_command",
    event_term_name: str = "push_robot",
    reward_term_name: str = "track_squat_pos",
) -> torch.Tensor:

    command_term: commands.StompCommand = env.command_manager.get_term(command_term_name)

    action_term: EventTermCfg = env.event_manager.get_term_cfg(event_term_name)
    ranges: dict = action_term.params["velocity_range"]
    max_range: dict = action_term.params["max_velocity_range"]
    speed: float = action_term.params["speed"]

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.85 and command_term.average_episode_length > env.max_episode_length * 0.96:

            for key in ranges:
                org = ranges[key]
                max_org = max_range[key]

                new_min = org[0] * speed
                new_max = org[1] * speed

                new_min = max(new_min, max_org[0])
                new_max = min(new_max, max_org[1])

                ranges[key] = (new_min, new_max)

    return torch.tensor(ranges["x"][1], device=env.device)

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from . import command_squat
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def squat_command(env: ManagerBasedRLEnv, command_name: str = "squat_command") -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return command


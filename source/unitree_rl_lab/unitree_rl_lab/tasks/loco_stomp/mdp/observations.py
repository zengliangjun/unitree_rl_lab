from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from unitree_rl_lab.tasks.loco_stomp.mdp import commands

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def gait_phase(env: ManagerBasedRLEnv, command_name="stomp_command") -> torch.Tensor:

    cmd: commands.StompCommand = env.command_manager.get_term(command_name)

    global_phase = cmd.feet_global_phases
    swing_phase = cmd.feet_swing_phases

    sin_phase = torch.sin(global_phase * torch.pi * 2.0)
    cos_phase = torch.cos(global_phase * torch.pi * 2.0)

    swing_sin_phase = torch.sin(swing_phase * torch.pi)
    swing_cos_phase = torch.cos(swing_phase * torch.pi)

    return torch.cat([sin_phase, cos_phase, swing_sin_phase, swing_cos_phase], dim=-1)

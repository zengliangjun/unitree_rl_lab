
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.assets import Articulation
import isaaclab.utils.string as string_utils

import pickle
import os

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .actions_cfg import MotionsActionCfg

class MotionActions(ActionTerm):

    cfg: MotionsActionCfg
    _asset: Articulation

    def __init__(self, cfg: MotionsActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, preserve_order=True)
        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        # parse scale


        if isinstance(cfg.scale, (float, int)):
            self._scale = torch.ones((len(self._joint_ids)), dtype=torch.float32)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")

        ## sdk index
        sdk_names = []
        for name in  env.cfg.scene.robot.joint_sdk_names:
            if 0 == len(name):
                continue
            sdk_names.append(name)

        sdk_indexes = []
        for name in cfg.joint_names:
            sdk_indexes.append(sdk_names.index(name))

        self.motions_dof_pos = []

        episode_length = int(env.cfg.episode_length_s * (1 / env.step_dt)) + 10

        for dirpath, _, files in os.walk(cfg.motions_dir):
            for filename in files:
                if not filename.endswith('.pkl'):
                    continue

                full_name = os.path.join(dirpath, filename)
                with open(full_name, "rb") as fd:
                    motions = pickle.load(fd)
                    motions = motions["dof_pos"][:, sdk_indexes]
                    motions = torch.tensor(motions, dtype=torch.float32, device=env.device)
                    self.motions_dof_pos.append(motions)

        self.motions_dof_pos = torch.cat(self.motions_dof_pos, dim=0)
        self.max_idx = self.motions_dof_pos.shape[0] - episode_length
        self.current_idx = torch.zeros((self.num_envs), dtype=torch.long, device=env.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    """
    Operations.
    """
    def process_actions(self, actions: torch.Tensor):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
            count = self.num_envs
        else:
            count = len(env_ids)

        selected_idxs = torch.randint(high=self.max_idx, size=(count,), dtype=torch.long, device=self._env.device)
        self.current_idx[env_ids] = selected_idxs

    def apply_actions(self):
        pos = self.motions_dof_pos[self.current_idx]
        self._raw_actions[:] = pos
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
        self.current_idx += 1

        reset_flags = self.motions_dof_pos >= (self.motions_dof_pos.shape[0] -1)
        if torch.sum(reset_flags) > 0:
            self.current_idx[reset_flags] = 0

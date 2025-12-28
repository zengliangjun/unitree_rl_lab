
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

        ##
        self.motions_max_idx = []
        self.motions_dof_pos = []

        episode_length = int(env.cfg.episode_length_s * (1 / env.step_dt)) + 10

        for dirpath, _, files in os.walk(cfg.motions_dir):
            for filename in files:
                if not filename.endswith('.pkl'):
                    continue

                full_name = os.path.join(dirpath, filename)
                with open(full_name, "rb") as fd:
                    motions = pickle.load(fd)
                    motions = motions["dof_pos"]
                    #if motions.shape[0] < episode_length:
                    #    continue

                    self.motions_dof_pos.append(motions)
                    self.motions_max_idx.append(motions.shape[0])

        self.motions_max_idx = torch.tensor(self.motions_max_idx, dtype = torch.float32)
        self.motions_probs = self.motions_max_idx.float() / torch.sum(self.motions_max_idx)

        sdk_names = []
        for name in  env.cfg.scene.robot.joint_sdk_names:
            if 0 == len(name):
                continue
            sdk_names.append(name)

        self.sdk_indexes = []
        for name in cfg.joint_names:
            self.sdk_indexes.append(sdk_names.index(name))

        self.current_idx = torch.zeros((self.num_envs), dtype=torch.long)
        self.action_motion_idx = torch.zeros((self.num_envs), dtype=torch.long)
        self.motion_dir = torch.zeros((self.num_envs), dtype=torch.int)


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

        selected_idxs = torch.multinomial(self.motions_probs, num_samples=count, replacement=True)
        self.action_motion_idx[env_ids] = selected_idxs
        max_idx = self.motions_max_idx[selected_idxs]
        self.current_idx[env_ids] = torch.tensor(torch.rand_like(selected_idxs, dtype = torch.float32) * max_idx, dtype= torch.long)
        self.motion_dir[env_ids] = 1

    def apply_actions(self):
        poses = []
        for mid, indx in zip(self.action_motion_idx, self.current_idx):
            dof_pos = self.motions_dof_pos[mid]
            mid, indx = mid.item(), indx.item()

            dof_pos = torch.tensor(dof_pos[indx: indx + 1][:, self.sdk_indexes], dtype=torch.float32)
            poses.append(dof_pos)

        self.current_idx += self.motion_dir[:]
        over_min = self.current_idx < 0
        self.current_idx[over_min] = 1
        self.motion_dir[over_min] = 1

        max_idx = self.motions_max_idx[self.action_motion_idx]
        over_max = (self.current_idx - max_idx) >= 0
        self.current_idx[over_max] = self.current_idx[over_max] - 1
        self.motion_dir[over_min] = - 1

        pos = torch.cat(poses, dim=0)
        self._raw_actions[ :] = pos.to(self._env.device)
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

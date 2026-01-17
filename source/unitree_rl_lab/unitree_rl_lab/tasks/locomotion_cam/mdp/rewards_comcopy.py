from __future__ import annotations
from collections.abc import Sequence

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ManagerTermBase

import isaaclab.utils.math as math_utils
from . import cam_utils


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class CAM(ManagerTermBase):

    _env: ManagerBasedRLEnv

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

        self.hat_q_dot = torch.zeros((env.num_envs, asset.num_joints + 6), device=env.device)  # (B, N_DOF)
        self.asset = asset

        # 预计算不变的值
        self._precomputed_masses = None
        self._precomputed_inertias = None
        self._device = env.device

        # 延迟预计算：在第一次调用 calcute_2 时进行

    def _precompute_constants(self):
        """
        预计算不变的值

        该方法在第一次调用 calcute_2 时执行，预计算连杆质量和惯性张量。
        这些值在机器人配置不变的情况下是常量，可以避免每次调用时的重复计算。
        """
        if self._precomputed_masses is None:
            # 获取连杆质量
            self._precomputed_masses = self.asset.root_physx_view.get_masses().to(self._device)

            # 获取并 reshape 惯性张量
            inertias = self.asset.root_physx_view.get_inertias().to(self._device)
            self._precomputed_inertias = inertias.view(*inertias.shape[:2], 3, 3)

    def _calcute_com(self) -> torch.Tensor:
        dofs = self.asset.num_joints
        mass_matrices = self.asset.root_physx_view.get_generalized_mass_matrices()

        com_pos_w = self.asset.data.root_com_pos_w  # Centroidal Momentum Matrix (CMM) is (6, 6+nj) projected in CoM frame aligned with world frame
        rotation_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(self.asset.data.root_quat_w))
        rotation_matrix = rotation_matrix.repeat(self.num_envs, 1, 1)
        adjoint_matrix_twist = cam_utils.adjoint_matrix_twist(rotation_matrix, com_pos_w)

        block_adjoint_matrix_twist = torch.zeros(self.num_envs, dofs + 6, dofs + 6, device=self._env.device)
        block_adjoint_matrix_twist[:, :6, :6] = adjoint_matrix_twist
        block_adjoint_matrix_twist[:, 6:, 6:] = torch.eye(dofs).expand(self.num_envs, dofs, dofs)
        CoM_mass_matrix = block_adjoint_matrix_twist.permute(0,2,1) @ mass_matrices
        return CoM_mass_matrix

    def calcute_momentum_hat(self, CoM_mass_matrix: torch.Tensor, command_name: str) -> torch.Tensor:
        command = self._env.command_manager.get_command(command_name)
        self.hat_q_dot[:, 2] = command[:, 2]
        self.hat_q_dot[:, 3] = command[:, 0]
        self.hat_q_dot[:, 4] = command[:, 1]

        CMM = CoM_mass_matrix[:,:6,:] # A(q)
        CAM = torch.einsum("bij,bj->bi", CMM, self.hat_q_dot)  # (B, 6)
        return CAM

    def calcute_momentum_with_mass_matrix(self, CoM_mass_matrix: torch.Tensor) -> torch.Tensor:
        gen_vel_b = torch.hstack((self.asset.data.root_ang_vel_b,
                                self.asset.data.root_ang_vel_b,
                                self.asset.data.joint_vel)).unsqueeze(2)

        CMM = CoM_mass_matrix[:,:6,:] # A(q)
        CAM = (CMM @ gen_vel_b).squeeze(2) # A(q) * qdot
        return CAM

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        left_leg_names: list[str],
        right_leg_names: list[str],
        left_arm_names: list[str],
        right_arm_names: list[str],
        body_names: list[str],
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ) -> torch.Tensor:

        CoM_mass_matrix = self._calcute_com()
        _mass = self.calcute_momentum_with_mass_matrix(CoM_mass_matrix)
        _mass_hat = self.calcute_momentum_hat(CoM_mass_matrix, command_name)

        print("    diff1:", _mass_hat - _mass)
        return torch.zeros(env.num_envs, device=env.device)

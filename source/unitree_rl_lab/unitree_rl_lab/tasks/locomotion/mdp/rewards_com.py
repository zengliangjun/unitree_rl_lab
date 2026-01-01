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
        left_leg_names: list[str] = cfg.params["left_leg_names"] # pyright: ignore[reportAssignmentType]
        self.left_leg_ids =asset.find_bodies(left_leg_names, preserve_order=True)[0]

        right_leg_names: list[str] = cfg.params["right_leg_names"] # pyright: ignore[reportAssignmentType]
        self.right_leg_ids =asset.find_bodies(right_leg_names, preserve_order=True)[0]

        left_arm_names: list[str] = cfg.params["left_arm_names"] # pyright: ignore[reportAssignmentType]
        self.left_arm_ids =asset.find_bodies(left_arm_names, preserve_order=True)[0]

        right_arm_names: list[str] = cfg.params["right_arm_names"] # pyright: ignore[reportAssignmentType]
        self.right_arm_ids =asset.find_bodies(right_arm_names, preserve_order=True)[0]

        body_names: list[str] = cfg.params["body_names"] # pyright: ignore[reportAssignmentType]
        self.body_ids =asset.find_bodies(body_names, preserve_order=True)[0]

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

    def calcute_momentum_with_link(self):
        self._precompute_constants()

        # 获取质心状态
        com_pos = math_utils.quat_apply_inverse(self.asset.data.root_quat_w, self.asset.data.root_com_pos_w)
        com_vel = self.asset.data.root_com_lin_vel_b
        angle_vel = self.asset.data.root_com_ang_vel_b

        # 使用预计算的连杆质量和惯性张量
        link_masses = self._precomputed_masses
        link_inertia = self._precomputed_inertias

        # 获取连杆位置和速度（这些值随时间变化，不能预计算）
        link_pos = self.asset.data.body_com_pos_b
        link_vel = math_utils.quat_apply_inverse(self.asset.data.root_quat_w, self.asset.data.body_com_lin_vel_w)
        link_ang_vel = math_utils.quat_apply_inverse(self.asset.data.root_quat_w, self.asset.data.body_com_ang_vel_w)

        # 计算相对量（向量化操作）
        link_pos_rel = link_pos - com_pos.unsqueeze(1)
        link_vel_rel = link_vel - com_vel.unsqueeze(1)
        # link_ang_rel = link_ang_vel - angle_vel.unsqueeze(1)

        # 先计算质量加权的速度
        weighted_vel = link_vel_rel * link_masses.unsqueeze(2)
        # 再计算质心动量
        centroidal_momentum = torch.cross(link_pos_rel, weighted_vel, dim=-1)
        # 计算角动量
        centroidal_angle_momentum = torch.einsum("bnij,bnj->bni", link_inertia, link_ang_vel)

        momentum = torch.cat([centroidal_angle_momentum, centroidal_momentum], dim=-1)
        return torch.sum(momentum, dim =1)  # 在连杆维度上求和

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

        _link = self.calcute_momentum_with_link()

        CoM_mass_matrix = self._calcute_com()
        _mass = self.calcute_momentum_with_mass_matrix(CoM_mass_matrix)
        _mass_hat = self.calcute_momentum_hat(CoM_mass_matrix, command_name)

        print("CAM diff0:", _link - _mass)
        print("    diff1:", _mass_hat - _mass)
        print("    diff1:", _link - _mass_hat)
        return torch.zeros(env.num_envs, device=env.device)

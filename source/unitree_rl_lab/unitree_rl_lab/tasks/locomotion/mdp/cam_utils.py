# cam_utils.py
from __future__ import annotations
import torch
import isaaclab.utils.math as math_utils

from functools import singledispatch
import numpy as np

@singledispatch
def skew_symmetric(p):
    raise TypeError("Unsupported type! Only torch.Tensor and np.ndarray are supported.")

@skew_symmetric.register
def _(p: torch.Tensor) -> torch.Tensor:
    assert p.shape[-1] == 3, "Last dimension of input must be 3"

    # Ensure p has shape (num_envs, 3)
    if p.ndim == 1:
        p = p.unsqueeze(0)  # Convert (3,) -> (1, 3)

    zero = torch.zeros_like(p[..., 0])  # (num_envs,)
    px, py, pz = p[..., 0], p[..., 1], p[..., 2]  # Extract components

    # Construct batch of skew-symmetric matrices
    skew_matrices = torch.stack([
        torch.stack([zero, -pz, py], dim=-1),
        torch.stack([pz, zero, -px], dim=-1),
        torch.stack([-py, px, zero], dim=-1)
    ], dim=-2)  # Final shape (num_envs, 3, 3)

    return skew_matrices

@skew_symmetric.register
def _(p: np.ndarray) -> np.ndarray:
    assert p.shape[-1] == 3, "Last dimension of input must be 3"

    # Ensure p has shape (num_envs, 3)
    if p.ndim == 1:
        p = np.expand_dims(p, axis=0)  # Convert (3,) -> (1, 3)

    zero = np.zeros_like(p[..., 0])  # (num_envs,)
    px, py, pz = p[..., 0], p[..., 1], p[..., 2]  # Extract components

    # Construct batch of skew-symmetric matrices
    skew_matrices = np.stack([
        np.stack([zero, -pz, py], axis=-1),
        np.stack([pz, zero, -px], axis=-1),
        np.stack([-py, px, zero], axis=-1)
    ], axis=-2)  # Final shape (num_envs, 3, 3)

    return skew_matrices


def adjoint_matrix_twist(R: torch.Tensor, p_w: torch.Tensor) -> torch.Tensor:
    """Adjoint for twists: [R, 0; [p]x R, R] (world-aligned CoM frame)."""
    # R: (B,3,3), p_w: (B,3)
    B = R.shape[0]
    px = skew_symmetric(p_w)  # (B,3,3)
    upper = torch.cat([R, torch.zeros(B, 3, 3, device=R.device)], dim=-1)
    lower = torch.cat([px @ R, R], dim=-1)
    return torch.cat([upper, lower], dim=-2)  # (B,6,6)


def compute_com_mass_matrix(asset, env) -> torch.Tensor:
    """
    Build CoM mass matrix in CoM frame aligned with world.
    Returns: (B, 6+ndof, 6+ndof)
    """
    B = env.num_envs
    ndof = asset.num_joints

    M = asset.root_physx_view.get_generalized_mass_matrices()  # (B, 6+ndof, 6+ndof)
    com_pos_w = asset.data.root_com_pos_w  # (B,3)

    # world-aligned CoM frame: rotate base->world alignment using root quat
    R = math_utils.matrix_from_quat(math_utils.quat_inv(asset.data.root_quat_w))  # (B,3,3)
    Ad = adjoint_matrix_twist(R, com_pos_w)  # (B,6,6)

    T = torch.zeros(B, 6 + ndof, 6 + ndof, device=env.device)
    T[:, :6, :6] = Ad
    T[:, 6:, 6:] = torch.eye(ndof, device=env.device).expand(B, ndof, ndof)

    # Transform (note: consistent with your original left-multiply; keep it stable)
    M_com = T.transpose(1, 2) @ M
    return M_com


def get_cmm(M_com: torch.Tensor) -> torch.Tensor:
    """Centroidal momentum matrix block A_G(q): (B,6,6+ndof)."""
    return M_com[:, :6, :]


def embed_command_qdot(command_vxyz: torch.Tensor, ndof: int, device: torch.device) -> torch.Tensor:
    """
    Paper Eq.(7): qhat_dot = [0,0, w_hat_z, v_hat_x, v_hat_y, 0, 0_leg, 0_arm]
    command is Isaac Lab base command: [v_x, v_y, w_z] (B,3) or [vx,vy,wz,...] -> slice outside.
    Returns: (B, 6+ndof)
    """
    B = command_vxyz.shape[0]
    qhat = torch.zeros(B, 6 + ndof, device=device)

    # base twist order: [wx, wy, wz, vx, vy, vz]
    qhat[:, 2] = command_vxyz[:, 2]  # wz
    qhat[:, 3] = command_vxyz[:, 0]  # vx
    qhat[:, 4] = command_vxyz[:, 1]  # vy
    return qhat


def get_generalized_qdot(asset) -> torch.Tensor:
    """
    Actual generalized velocity qdot in base frame:
      [wx, wy, wz, vx, vy, vz, joint_vel...]
    """
    return torch.cat(
        [
            asset.data.root_ang_vel_b,  # (B,3)
            asset.data.root_lin_vel_b,  # (B,3)  <-- FIX vs your code
            asset.data.joint_vel,       # (B,ndof)
        ],
        dim=-1,
    )


def compute_cam(cmm: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
    """
    h = A(q) qdot ; returns centroidal momentum h (B,6).
    CAM is h[:,:3], CLM is h[:,3:].
    """
    return torch.einsum("bij,bj->bi", cmm, qdot)


def compute_cam_ref(cmm: torch.Tensor, command_vxyz: torch.Tensor, ndof: int, device) -> torch.Tensor:
    qhat = embed_command_qdot(command_vxyz, ndof=ndof, device=device)
    return compute_cam(cmm, qhat)  # (B,6)


def cam_dot(prev_cam: torch.Tensor, cam: torch.Tensor, dt: float) -> torch.Tensor:
    return (cam - prev_cam) / dt

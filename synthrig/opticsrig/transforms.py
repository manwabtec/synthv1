"""Rigid transforms. Naming: T_a_b maps points in frame b to frame a (x_a = T_a_b @ x_b).

All device frames are OpenCV-style (x right, y down, z forward). The ONLY place that
knows about Blender's camera/light convention (looks down -Z, +Y up) is
`CV_TO_BLENDER_OBJ` below, used by the Blender backend.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .schema import Pose

# Blender cameras/lights look down their local -Z with +Y up. An OpenCV device frame
# with the same pose has its z flipped and y flipped:  R_blender = R_cv @ diag(1,-1,-1).
CV_TO_BLENDER_OBJ = np.diag([1.0, -1.0, -1.0, 1.0])


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return X @ T[:3, :3].T + T[:3, 3]


def look_at_R(position, target, up) -> np.ndarray:
    """Rotation whose z axis points from position to target; y axis is 'down' (opposite `up`)."""
    z = np.asarray(target, float) - np.asarray(position, float)
    n = np.linalg.norm(z)
    if n < 1e-12:
        raise ValueError("look_at target equals position")
    z /= n
    up = np.asarray(up, float)
    x = np.cross(-up, z)  # y_down = -up  ->  x = y × z = (-up) × z
    if np.linalg.norm(x) < 1e-9:
        raise ValueError("look_at: up vector parallel to viewing direction")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def pose_to_T(pose: Pose) -> np.ndarray:
    """Device -> rig transform from a schema Pose."""
    p = np.asarray(pose.position, float)
    if pose.look_at is not None:
        R = look_at_R(p, pose.look_at, pose.up)
    elif pose.euler_xyz_deg is not None:
        R = Rotation.from_euler("xyz", pose.euler_xyz_deg, degrees=True).as_matrix()
    elif pose.R is not None:
        R = np.asarray(pose.R, float)
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or np.linalg.det(R) < 0:
            raise ValueError("pose.R is not a proper rotation matrix")
    else:
        R = np.eye(3)
    return make_T(R, p)


def rig_pose_look_at(position, target, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """T_world_rig placing an OpenCV-style rig frame at `position` looking at `target`
    in a Z-up world (Blender's world). Handy for scenes."""
    return make_T(look_at_R(position, target, world_up), position)

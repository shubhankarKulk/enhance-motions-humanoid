import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import math
import mujoco
from utils.transformUtils import *

def get_heading_q(q):
    hq = q.copy()
    hq[1] = 0
    hq[2] = 0
    hq /= np.linalg.norm(hq)
    return hq

def getHeading(q):
    hq = q.copy()
    hq[1] = 0
    hq[2] = 0
    if hq[3] < 0:
        hq *= -1
    hq /= np.linalg.norm(hq)
    return 2 * math.acos(hq[0])

def getHeadingQ(q):
    hq = np.array([0.0, 0.0, q[3], q[2]])  # z, w components
    norm = np.linalg.norm(hq)
    if norm < 1e-10:  # Avoid division by zero
        return np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    hq /= norm
    return hq

def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm

def get_qvel(cur_qpos, next_qpos, dt):
    v = (next_qpos[:3] - cur_qpos[:3]) / dt
    qrel = quaternion_multiply(next_qpos[3:7], quaternion_inverse(cur_qpos[3:7]))
    axis, angle = rotation_from_quaternion(qrel, True)
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    rv = (axis * angle) / dt
    rv = transformVec(rv, cur_qpos[3:7])   # angular velocity is in root coord
    diff = next_qpos[7:] - cur_qpos[7:]
    while np.any(diff > np.pi):
        diff[diff > np.pi] -= 2 * np.pi
    while np.any(diff < -np.pi):
        diff[diff < -np.pi] += 2 * np.pi
    qvel = diff / dt
    qvel = np.concatenate((v, rv, qvel))
    v = get_quaternion_heading(v, cur_qpos[3:7])
    qvel[:3] = v
    return qvel

def get_angvel(prev_bquat, cur_bquat, dt):
    q_diff = multi_quat_diff(cur_bquat, prev_bquat)
    n_joint = q_diff.shape[0] // 4
    body_angvel = np.zeros(n_joint * 3)
    for i in range(n_joint):
        body_angvel[3*i: 3*i + 3] = rotation_from_quaternion(q_diff[4*i: 4*i + 4]) / dt
    return body_angvel

def get_quaternion_headingEnv(v, q):
    """Transform vector v to local frame by removing yaw from quaternion q."""
    hq = q.copy()
    hq[1] = 0  # Remove x (pitch/roll)
    hq[2] = 0  # Remove y (pitch/roll)
    norm = np.linalg.norm(hq)
    if norm < 1e-8:
        return v.copy()  # No rotation if quaternion is near zero
    hq /= norm
    rot = quatMat(hq)
    return rot.T @ v

def get_quaternion_heading(v, q):
    hq = q.copy()
    hq[1] = 0
    hq[2] = 0
    hq /= np.linalg.norm(hq)
    rot = quaternion_matrix(hq)[:3, :3]
    v = rot.T.dot(v[:, None]).ravel()
    return v

def deHeading(q):
    return quatMul(quatInv(getHeadingQ(q)), q)

def quatMul(quaternion1, quaternion0):
    """Return multiplication of two quaternions.

    >>> q = quaternion_multiply([4, 1, -2, 3], [8, -5, 6, 7])
    >>> numpy.allclose(q, [28, -44, -14, 48])
    True

    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([
        -x1*x0 - y1*y0 - z1*z0 + w1*w0,
        x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
        x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)
    
def quatInv(quaternion):
    """Return inverse of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_inverse(q0)
    >>> numpy.allclose(quaternion_multiply(q0, q1), [1, 0, 0, 0])
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q / np.dot(q, q)

def transformVec(v, q):
    rot = quaternion_matrix(q)[:3, :3]
    v = rot.T.dot(v[:, None]).ravel()
    return v


def multi_quat_norm(nq):
    """return the scalar rotation of a N joints"""

    nq_norm = np.arccos(np.clip(abs(nq[::4]), -1.0, 1.0))
    return nq_norm


def multi_quat_diff(nq1, nq0):
    """return the relative quaternions q1-q0 of N joints"""

    nq_diff = np.zeros_like(nq0)
    for i in range(nq1.shape[0] // 4):
        ind = slice(4*i, 4*i + 4)
        q1 = nq1[ind]
        q0 = nq0[ind]
        nq_diff[ind] = quatMul(q1, quatInv(q0))
    return nq_diff

def quatMat(q):
    """Converts a quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

def get_body_qpos_addr(model):
    body_qposaddr = dict()
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]
    for i, body_name in enumerate(body_names):
        start_joint = model.body_jntadr[i]
        if start_joint < 0:
            continue
        end_joint = start_joint + model.body_jntnum[i]
        start_qposaddr = model.jnt_qposadr[start_joint]
        if end_joint < len(model.jnt_qposadr):
            end_qposaddr = model.jnt_qposadr[end_joint]
        else:
            end_qposaddr = model.nq
        body_qposaddr[body_name] = (start_qposaddr, end_qposaddr)
    return body_qposaddr

def compute_pose_reward(current_joint_pos, expert_joint_pos, scale=2.0):
    pose_diff = multi_quat_norm(multi_quat_diff(current_joint_pos[4:], expert_joint_pos[4:]))
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = np.exp(-scale * (pose_dist ** 2))
    return pose_reward

def compute_velocity_reward(current_qvel, expert_qvel, scale=0.005):
    vel_dist = np.linalg.norm(current_qvel[3:] - expert_qvel[3:])  # ignore root
    vel_reward = math.exp(-scale * (vel_dist ** 2))
    return vel_reward

def compute_com_reward(current_com, expert_com, scale=10.0):
    com_error = np.mean((current_com - expert_com) ** 2)
    return np.exp(-scale * com_error)

def compute_end_effector_reward(cur_ee, e_ee, scale=40):
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-scale * (ee_dist ** 2))
    return ee_reward

def compute_height_reward(cur_qpos, e_qpos, cur_rq_rmh, e_rq_rmh, scale=300.0):
    root_height_dist = cur_qpos[2] - e_qpos[2]
    root_quat_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]
    root_pose_reward = math.exp(-scale * (root_height_dist ** 2) - scale * (root_quat_dist ** 2))
    return root_pose_reward

def compute_root_vel_reward(cur_rlinv_local, e_rlinv_local, cur_rangv, e_rangv, scale_1=1.0, scale_2=0.1):
    root_linv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)
    root_angv_dist = np.linalg.norm(cur_rangv - e_rangv)
    root_vel_reward = math.exp(-scale_1 * (root_linv_dist ** 2) - scale_2 * (root_angv_dist ** 2))
    return root_vel_reward

def pose_reward(current_joint_pos, expert_joint_pos, scale=10.0):
    error_sum = 0.0
    for name in current_joint_pos:
        rel_current = current_joint_pos[name] - current_joint_pos["root"]
        rel_expert = expert_joint_pos[name] - expert_joint_pos["root"]
        error_sum += np.linalg.norm(rel_current - rel_expert)
    normalized_error = error_sum / (len(current_joint_pos) - 1)  # excluding root
    return np.exp(-scale * normalized_error)

def velocity_reward(current_qvel, expert_qvel, scale=0.1):
    vel_error = np.mean((current_qvel - expert_qvel) ** 2)
    return np.exp(-scale * vel_error)

def com_reward(current_com, expert_com, scale=10.0):
    com_error = np.mean((current_com - expert_com) ** 2)
    return np.exp(-scale * com_error)

def com_height_reward(current_com, expert_com, scale=5.0):
    height_error = abs(current_com - expert_com)  # z is vertical
    return np.exp(-scale * height_error)

def get_center_of_mass(model, data):
    total_mass = 0.0
    com = np.zeros(3)
    for i in range(model.nbody):
        mass = model.body_mass[i]
        pos = data.xipos[i]
        com += mass * pos
        total_mass += mass
    return com / total_mass

def quat_to_mat(quat):
    """Convert quaternion (x,y,z,w) to 3x3 rotation matrix"""
    x, y, z, w = quat
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * y * y - 2 * z * z
    R[0, 1] = 2 * x * y - 2 * z * w
    R[0, 2] = 2 * x * z + 2 * y * w
    R[1, 0] = 2 * x * y + 2 * z * w
    R[1, 1] = 1 - 2 * x * x - 2 * z * z
    R[1, 2] = 2 * y * z - 2 * x * w
    R[2, 0] = 2 * x * z - 2 * y * w
    R[2, 1] = 2 * y * z + 2 * x * w
    R[2, 2] = 1 - 2 * x * x - 2 * y * y
    return R

def rotate_vecs(vecs, R):
    """Rotate vectors into root frame"""
    return vecs @ R.T

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def set_mean_std(self, mean, std, n):
        self.rs._n = n
        self.rs._M[...] = mean
        self.rs._S[...] = std
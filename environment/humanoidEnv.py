import numpy as np
import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium import spaces
from utils.envUtils import *
from utils.mocapTransform import MocapDM
from utils.mocapUtils import *
from utils.transformUtils import *
import mujoco
from scipy.linalg import cho_solve, cho_factor

class HumanoidEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }
    
    def __init__(self, modelPath, filePath, render_mode = None):
        self.frame_skip = 5
        MujocoEnv.__init__(self,
                        model_path=modelPath, 
                        frame_skip=self.frame_skip,
                        observation_space=None,
                        render_mode=render_mode)
        
        self.terminate_when_unhealthy = True
        self.action_dim = 28
        self.state_dim = 132
        low = np.full(self.action_dim, -1, dtype=np.float32)
        high = np.full(self.action_dim, 1, dtype=np.float32)

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float64)
        
        self.motion = MocapDM()
        self.idx_curr = -1
        self.expert = None
        
        self.body_qposaddr = get_body_qpos_addr(self.model)
        self.bquat = self.get_body_quat()
        self.prev_bquat = None
        self.r_height = 0.0
        self.imitation_r = 0.0

        self.load_mocap(filePath)
        self.start_ind = 0
        self.cur_t = 0
        self.rewards = [0, 0, 0, 0, 0, 0]

        self.joint_names = ['chest_x', 'chest_y', 'chest_z', 'neck_x', 'neck_y', 'neck_z',
                           'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 'right_elbow',
                           'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'left_elbow',
                           'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_knee',
                           'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
                           'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_knee',
                           'left_ankle_x', 'left_ankle_y', 'left_ankle_z']

        joint_ids = [self.model.joint(j).id for j in self.joint_names]
        self.joint_ranges = np.array([self.model.jnt_range[j] for j in joint_ids])

        self.calculate_kp_jd()

        self.expert_ee_sites = [
            "expert_right_ankle",
            "expert_left_ankle",
            "expert_right_wrist",
            "expert_left_wrist"
        ]

        self.body_names_to_check = [
            "chest", "neck", "right_elbow", "right_hip", "right_knee",
            "right_shoulder", "right_wrist", "left_elbow", "left_hip",
            "left_knee", "left_shoulder", "left_wrist", "right_ankle",
            "left_ankle"
        ]
        # Verify that the sites exist in the model
        for site_name in self.expert_ee_sites:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name) == -1:
                raise ValueError(f"Site '{site_name}' not found in the MuJoCo model. Check the XML file.")

        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.floor_geom_id == -1:
            raise ValueError("Floor geom not found in the model. Check XML file.")
        
        self.humanoid_com_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "humanoid_com")
        self.expert_com_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "expert_com")
        self.ee_site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s) for s in self.expert_ee_sites]
        self.ndof = self.model.actuator_ctrlrange.shape[0]
        self.pos_err = 0.0
        self.vel_err = 0.0
        self.max_torque = 0.0
        self.fixed_start = False
        self.max_distance = 0.0
        
        self.set_mode = None
        self.right_knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")
        self.left_knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")
        self.right_hip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip")
        self.left_hip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip")
        self.actuator_ids = [self.model.actuator(j).id for j in self.joint_names]
        self.max_expert_distance = self._compute_max_expert_distance()

    @property
    def is_healthy(self):
        min_z, max_z = [0.3, 2.5]
        # print(self.data.qpos[2])
        is_healthy = min_z < self.data.qpos[2] < max_z
        
        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self.terminate_when_unhealthy else False
        return terminated
    
    def calculate_kp_jd(self):
        self.kp = np.zeros(self.action_dim)
        self.kd = np.zeros(self.action_dim)
        for i, joint_name in enumerate(self.joint_names):
            if joint_name in PARAMS_KP_KD:
                kp, kd = PARAMS_KP_KD[joint_name]
                self.kp[i] = kp
                self.kd[i] = kd

    def get_phaseEval(self):
        phase = self.idx_curr / (self.mocap_data_len - 1)
        return np.clip(phase, 0.0, 1.0)
    
    def get_ee_index(self):
        return self.idx_curr % self.mocap_data_len
        
    def get_phase(self):
        ind = self.get_expert_index(self.cur_t)
        phase = ind / self.mocap_data_len
        return np.clip(phase, 0.0, 1.4)
    
    def get_expert_index(self, t):
        return self.start_ind + t
    
    def get_body_quat(self):
        qpos = self.data.qpos.copy()
        body_quat = [qpos[3:7]]
        body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)]
        for body in body_names[1:]:
            if body == 'root' or not body in self.body_qposaddr:
                continue
            start, end = self.body_qposaddr[body]
            euler = np.zeros(3)
            euler[:end - start] = qpos[start:end]
            quat = quaternion_from_euler(euler[0], euler[1], euler[2])
            body_quat.append(quat)
        body_quat = np.concatenate(body_quat)
        return body_quat
    
    def get_name_quat(self, name):
        qpos = self.data.qpos.copy()
        start, end = self.body_qposaddr[name]
        euler = np.zeros(3)
        euler[:end - start] = qpos[start:end]
        quat = quaternion_from_euler(euler[0], euler[1], euler[2])
        return quat

    def _get_obs(self):
        data = self.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        root_pos = qpos[:3].copy()
        root_quat = qpos[3:7].copy()
        root_mat = quatMat(root_quat)
        qvel[:3] = get_quaternion_headingEnv(qvel[:3], root_quat).ravel()
        qvel[3:6] = np.dot(root_mat.T, qvel[3:6])  # Root angular velocity in local frame
        
        # Relative positions
        rel_pos = []
        for i in range(1, self.model.nbody):
            pos = data.xpos[i].copy() - root_pos
            pos = np.dot(root_mat.T, pos)
            rel_pos.append(pos)
        rel_pos = np.array(rel_pos).flatten()
        
        # Body quaternions
        bquat = self.get_body_quat()
        
        # Joint velocities
        joint_vel = qvel[6:]
        
        # Phase
        phase = self.get_phase()
        
        obs = np.concatenate([
            rel_pos,           # Relative link positions
            bquat.flatten(),   # Body quaternions
            qvel[:3],          # Root linear velocity
            qvel[3:6],         # Root angular velocity
            joint_vel,         # Joint velocities
            [phase]            # Phase
        ])
        return obs
    
    def load_mocap(self, filepath):
        self.motion.load_mocap(filepath)
        self.mocap_dt = self.motion.dt
        self.mocap_data_len = len(self.motion.dataset)
        self.expert = self.motion.getExpert(self, self.motion.data_config, mode="noToes")

    def check_floor_contact(self):
        phase = self.get_phase()
        contacts = {}
        for body_name in self.body_names_to_check:
            contacts[body_name] = False
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                continue
            geom_ids = [geom_id for geom_id in range(self.model.ngeom) if self.model.geom_bodyid[geom_id] == body_id]
            
            contact_points = []
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if (contact.geom1 == self.floor_geom_id and contact.geom2 in geom_ids) or \
                (contact.geom2 == self.floor_geom_id and contact.geom1 in geom_ids):
                    contact_points.append(contact.pos)
            
            if body_name in ['right_ankle', 'left_ankle'] and phase >= 0.9:
                heel_site = f"heel_{body_name}"
                heel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, heel_site)
                if heel_id != -1:
                    heel_pos = self.data.site_xpos[heel_id]
                    heel_contact = any(np.linalg.norm(contact_pos - heel_pos) < 0.01 for contact_pos in contact_points)
                    contacts[body_name] = heel_contact
                else:
                    # Single contact point with orientation check
                    if len(contact_points) > 0:  # Check for at least one contact
                        foot_quat = self.get_name_quat(body_name)
                        z_axis = self.quat_to_z_axis(foot_quat)
                        if np.dot(z_axis, np.array([0, 0, 1])) > 0.95:  # Ensure foot is nearly vertical
                            contacts[body_name] = True
            else:
                contacts[body_name] = len(contact_points) > 0
        return contacts

    def quat_to_z_axis(self, quat):
        w, x, y, z = quat
        z_axis = np.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y)
        ])
        return z_axis / np.linalg.norm(z_axis)
    
    def get_com(self):
        return self.data.subtree_com[0, :].copy()

    def get_ee_pos(self):
        ee_name = ["right_ankle", "left_ankle", "right_wrist", "left_wrist"]
        ee_pos = []
        for name in ee_name:
            bone_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            bone_vec = self.data.xpos[bone_id]
            ee_pos.append(bone_vec)
        return np.concatenate(ee_pos)

    def get_angvel(self, prev_bquat, cur_bquat, dt):
        q_diff = multi_quat_diff(cur_bquat, prev_bquat)
        n_joint = q_diff.shape[0] // 4
        body_angvel = np.zeros(n_joint * 3)
        for i in range(n_joint):
            body_angvel[3*i: 3*i + 3] = rotation_from_quaternion(q_diff[4*i: 4*i + 4]) / dt
        return body_angvel
    
    def get_expert_attr(self, attr, ind):
        ind = min(ind, self.mocap_data_len - 1) 
        return self.expert[attr][ind]
         
    def get_foot_pos(self, foot_name):
        if foot_name not in ['left_ankle', 'right_ankle']:
            raise ValueError(f"Invalid foot name: {foot_name}. Use 'left_ankle' or 'right_ankle'.")
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"joint_{foot_name}")
        if site_id == -1:
            raise ValueError(f"Site '{foot_name}_site' not found in the model.")
        return self.data.site_xpos[site_id].copy()
    
    def reward(self):
        t = self.cur_t
        ind = self.get_expert_index(t)
        prev_bquat = self.prev_bquat
        phase = self.get_phase()

        if ind > self.mocap_data_len - 1:
            ind = self.mocap_data_len - 1

        if prev_bquat is None:
            prev_bquat = self.bquat.copy()
        
        # Current state
        cur_com = self.get_com()
        cur_bquat = self.get_body_quat()
        cur_bangvel = self.get_angvel(prev_bquat, cur_bquat, self.dt)
        cur_ee = self.get_ee_pos()

        # Reference state
        e_ee = self.get_expert_attr('ee_pos', ind)
        e_bquat = self.get_expert_attr('bquat', ind)
        e_bangvel = self.get_expert_attr('bangvel', ind)
        e_com = self.get_expert_attr('com', ind)

        # Imitation reward
        pose_r = compute_pose_reward(cur_bquat, e_bquat)
        vel_r = compute_velocity_reward(cur_bangvel[3:], e_bangvel[3:])
        ee_r = compute_end_effector_reward(cur_ee, e_ee)
        com_r = compute_com_reward(cur_com, e_com)

        cur_z = self.data.qpos[2]
        e_z = self.get_expert_attr('qpos', ind)[2]
        height_r = com_height_reward(cur_z, e_z)

        w_p, w_v, w_e, w_c, w_r = 0.5, 0.05, 0.15, 0.2, 0.1
        imitation_r = w_p * pose_r + w_v * vel_r + w_e * ee_r + w_c * com_r + w_r * height_r
        self.imitation_r = imitation_r

        # Task rewards
        cur_x = self.data.qpos[0]
        if not hasattr(self, 'jump_start_x'):
            self.jump_start_x = cur_x
        forward_distance = cur_x - self.jump_start_x
        self.max_distance = max(self.max_distance, forward_distance)

        # Foot symmetry
        left_foot = self.get_foot_pos('left_ankle')
        right_foot = self.get_foot_pos('right_ankle')
        foot_x_diff = abs(left_foot[0] - right_foot[0]) ** 2
        foot_z_diff = abs(left_foot[2] - right_foot[2]) ** 2
        foot_diff = np.sqrt(foot_x_diff + foot_z_diff)
        r_foot_symmetry = np.exp(-10.0 * foot_diff)

        # Lateral symmetry
        com_y = cur_com[1]
        r_lateral = np.exp(-10.0 * abs(com_y))

        # Combined symmetry reward
        r_symmetry = 0.6 * r_foot_symmetry + 0.4 * r_lateral

        # Contact check
        floor_contacts = self.check_floor_contact()
        contact_score = 1.0 if floor_contacts["right_ankle"] and floor_contacts["left_ankle"] else 0.3 if floor_contacts["right_ankle"] or floor_contacts["left_ankle"] else 0.01
        non_foot_contact = any(floor_contacts[name] for name in ['chest', 'neck', 'right_elbow', 'right_hip', 'right_knee',
                                                                'right_shoulder', 'right_wrist', 'left_elbow', 'left_hip',
                                                                'left_knee', 'left_shoulder', 'left_wrist'])
        if non_foot_contact:
            contact_score *= 0.2

        # Post-landing upright posture reward
        r_upright = 0.0
        if phase > 0.7:  # Post-landing phase
            # Torso alignment (vertical)
            torso_quat = self.get_name_quat('chest')
            z_axis = self.quat_to_z_axis(torso_quat)
            vertical_axis = np.array([0, 0, 1])  # World z-axis
            torso_error = np.arccos(np.clip(np.dot(z_axis, vertical_axis), -1.0, 1.0))
            r_torso = np.exp(-10.0 * torso_error)

            # COM velocity (ensure stillness)
            com_vel = self.data.qvel[:3]
            com_vel_norm = np.linalg.norm(com_vel)
            r_com_vel = np.exp(-7.0 * com_vel_norm)

            r_upright = 0.5 * r_torso + 0.3 * r_com_vel + 0.2 * contact_score

        # Distance reward
        r_distance = 0.0
        avg_foot_x = np.mean([left_foot[0], right_foot[0]])
        if 0.3 <= phase <= 0.7:
            # Forward distance (COM displacement)
            forward_distance = cur_x - self.jump_start_x

            # Foot distance (average foot x-position relative to start)
            avg_foot_x = np.mean([left_foot[0], right_foot[0]])
            foot_distance = avg_foot_x - self.jump_start_x

            # Velocity (forward and vertical)
            com_vel = self.data.qvel[:3]
            forward_vel = np.clip(com_vel[0], 0, 5.0) / 5.0
            vertical_vel = np.clip(com_vel[2], 0, 3.0) / 3.0
            vel_bonus = 0.5 * forward_vel + 0.5 * vertical_vel

            # Combined distance reward
            r_distance = 0.4 * forward_distance + 0.4 * foot_distance + 0.2 * vel_bonus

        r_yaw = 0.0
        root_quat = self.data.qpos[3:7]
        yaw_angle = self.quat_to_yaw(root_quat)
        yaw_error = abs(yaw_angle)
        
        r_yaw = np.exp(-10.0 * yaw_error) 

        # Curriculum learning
        curriculum_factor = min(1.0, self.episode / 10000000)

        # Dynamic weights

        if self.set_mode == "jumpMimicReset":
            w_S = 1.0
            w_dist = 0.0
            w_sym = 0.0
            w_upright = 0.0
            w_yaw = 0.0
        else:
            w_S = 0.5 * (1.0 - curriculum_factor) + 0.1
            w_dist = 0.5 * curriculum_factor + 0.3
            w_sym = 0.1 * curriculum_factor + 0.05
            w_upright = 0.3 * curriculum_factor + 0.1 if phase > 0.7 else 0.0  # Post-landing only
            w_yaw = 0.2 * (1.0 - curriculum_factor) + 0.05

        total_reward = (w_S * imitation_r +
                        w_dist * r_distance + w_sym * r_symmetry + w_upright * r_upright + w_yaw * r_yaw)
        
        self.rewards = [pose_r, vel_r, ee_r, com_r, height_r, r_distance, r_symmetry, r_upright]

        self.cur_t += 1
        self.idx_curr += 1
        self.idx_curr = self.idx_curr % self.mocap_data_len

        return total_reward

    def quat_to_yaw(self, quat):
        q0, q1, q2, q3 = quat
        yaw = math.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
        return yaw
    
    def quat_to_pitch(self, quat):
        q0, q1, q2, q3 = quat
        # Compute pitch angle (rotation around x-axis)
        pitch = math.asin(2.0 * (q0 * q2 - q3 * q1))
        return pitch

    def quat_to_roll(self, quat):
        q0, q1, q2, q3 = quat
        roll = math.atan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
        return roll
    
    def quat_to_z_axis(self, quat):
        # Convert quaternion to z-axis vector (for torso alignment)
        q0, q1, q2, q3 = quat
        z_axis = np.array([
            2 * (q1 * q3 - q0 * q2),
            2 * (q2 * q3 + q0 * q1),
            q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
        ])
        return z_axis / np.linalg.norm(z_axis)  # Normalize
    
    def _compute_max_expert_distance(self):
        max_dist = 0.0
        for i in range(self.mocap_data_len):
            com = self.get_expert_attr('com', i)
            if i == 0:
                start_x = com[0]
            max_dist = max(max_dist, com[0] - start_x)
        return max_dist

    def compute_torque(self, action):
        joint_names = [
            ('chest_x', 200, 1), ('chest_y', 200, 1), ('chest_z', 200, 1),
            ('neck_x', 50, 1), ('neck_y', 50, 1), ('neck_z', 50, 1),
            ('right_shoulder_x', 100, 1), ('right_shoulder_y', 100, 1), ('right_shoulder_z', 100, 1), ('right_elbow', 60, 5),
            ('left_shoulder_x', 100, 1), ('left_shoulder_y', 100, 1), ('left_shoulder_z', 100, 1), ('left_elbow', 60, 5),
            ('right_hip_x', 200, 1), ('right_hip_y',400, 5), ('right_hip_z', 200, 1), ('right_knee', 400, 5),
            ('right_ankle_x', 90, 1), ('right_ankle_y', 200, 5), ('right_ankle_z', 90, 1),
            ('left_hip_x', 200, 1), ('left_hip_y', 400, 5), ('left_hip_z', 200, 1), ('left_knee', 400, 5),
            ('left_ankle_x', 90, 1), ('left_ankle_y', 200, 5), ('left_ankle_z', 90, 1)
        ]
        
        joint_to_body = {
            'chest_x': 'chest', 'chest_y': 'chest', 'chest_z': 'chest',
            'neck_x': 'neck', 'neck_y': 'neck', 'neck_z': 'neck',
            'right_shoulder_x': 'right_shoulder', 'right_shoulder_y': 'right_shoulder', 'right_shoulder_z': 'right_shoulder',
            'right_elbow': 'right_elbow',
            'left_shoulder_x': 'left_shoulder', 'left_shoulder_y': 'left_shoulder', 'left_shoulder_z': 'left_shoulder',
            'left_elbow': 'left_elbow',
            'right_hip_x': 'right_hip', 'right_hip_y': 'right_hip', 'right_hip_z': 'right_hip',
            'right_knee': 'right_knee',
            'right_ankle_x': 'right_ankle', 'right_ankle_y': 'right_ankle', 'right_ankle_z': 'right_ankle',
            'left_hip_x': 'left_hip', 'left_hip_y': 'left_hip', 'left_hip_z': 'left_hip',
            'left_knee': 'left_knee',
            'left_ankle_x': 'left_ankle', 'left_ankle_y': 'left_ankle', 'left_ankle_z': 'left_ankle'
        }

        dt = self.model.opt.timestep
        ctrl_joint = action  # Shape: (28,)
        qpos = self.data.qpos.copy()  # Shape: (39,) [7 free joint + 32 actuated]
        qvel = self.data.qvel.copy()  # Shape: (38,) [6 free joint + 32 actuated]
        
        k_p = np.zeros(self.ndof)
        k_d = np.zeros(self.ndof)
        gear = np.zeros(self.ndof)
        
        ind = self.get_expert_index(self.cur_t)
        if ind > self.mocap_data_len - 1:
            ind = self.mocap_data_len - 1
        base_pose = self.motion.data_config[ind][7:]  # Current expert pose
        target_pos = np.zeros(self.ndof)
        phase = self.get_phase()
        
        for i, (joint_name, gears, scale) in enumerate(joint_names):
            body_part = joint_to_body[joint_name]
            if joint_name in ["right_knee", "left_knee", "right_hip_y", "left_hip_y"]:
                k_p[i] = PARAMS_KP_KD[body_part][0] * 0.5
                k_d[i] = PARAMS_KP_KD[body_part][1] * 0.5
            else:
                k_p[i] = PARAMS_KP_KD[body_part][0] * 1.0
                k_d[i] = PARAMS_KP_KD[body_part][1] * 2.0
            gear[i] = gears
            target_pos[i] = base_pose[i] + ctrl_joint[i] * scale
        
        qpos_err = np.zeros(self.model.nv)
        qvel_err = np.zeros(self.model.nv)

        # qpos_err[6:] = qpos[7:] - target_pos
        qpos_err[6:] = qpos[7:] + qvel[6:] * dt - target_pos
        qvel_err = qvel
        q_accel = self.compute_desired_accel(qpos_err, qvel_err, k_p, k_d)
        qvel_err += q_accel * dt
        controls = -k_p * qpos_err[6:] - k_d * qvel_err[6:]
        controls /= gear
        controls = np.clip(controls, -1, 1)
        # print(f"Target pos: {np.array(controls[self.right_knee_id])}")

        self.max_torque = np.max(np.abs(controls * gear))
        self.pos_err = np.linalg.norm(qpos_err[6:])
        self.vel_err = np.linalg.norm(qvel_err[6:])
        return controls

    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d):

        dt = self.model.opt.timestep
        nv = self.model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        C = self.data.qfrc_bias.copy()
        
        K_p = np.diag(np.concatenate([np.zeros(6), k_p]))
        K_d = np.diag(np.concatenate([np.zeros(6), k_d]))
        
        q_accel = cho_solve(
            cho_factor(M + K_d * dt, overwrite_a=True, check_finite=False),
            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]),
            overwrite_b=True,
            check_finite=False
        )
        return q_accel.squeeze()
    
    def step(self, action):
        self.prev_bquat = self.bquat.copy()

        action = self.compute_torque(action)
        self.do_simulation(action, self.frame_skip)
        self.bquat = self.get_body_quat()

        # ind = self.get_ee_index()
        ind = self.get_expert_index(self.cur_t + self.start_ind)
        if ind > self.mocap_data_len - 1:
            ind = self.mocap_data_len - 1
        expert_ee_pos = self.get_expert_attr('ee_pos', ind)

        for i, site_name in enumerate(self.expert_ee_sites):
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            self.data.site_xpos[site_id] = expert_ee_pos[i*3:(i+1)*3]

        humanoid_com = self.get_com()  # Humanoid CoM
        expert_com = self.get_expert_attr('com', ind)  # Expert CoM
        self.data.site_xpos[self.humanoid_com_site_id] = humanoid_com
        self.data.site_xpos[self.expert_com_site_id] = expert_com

        reward = self.reward()

        observation = self._get_obs()
        if self.render_mode == "human":
            self.render()

        truncated = False
        info = {
            'reward_task': self.imitation_r,
            'reward_style': self.r_height
        }
        
        done = self.terminated
        floor_contacts = self.check_floor_contact()
        critical_contacts = any(floor_contacts[name] for name in ['chest', 'neck', "right_elbow", "right_hip", "right_knee",
                                                                    "right_shoulder", "right_wrist", "left_elbow", "left_hip",
                                                                    "left_knee", "left_shoulder", "left_wrist"])
        # end = self.cur_t + self.start_ind >= self.mocap_data_len
        end = self.get_phase() >= 1.4
        done = self.terminated or critical_contacts or end
        if self.terminated or critical_contacts:
            reward -= 1.0
        return observation, reward, done, truncated, info
        
    def reset_model(self):
        self.episode = getattr(self, 'episode', 0) + 1
        ind = 0 if self.fixed_start else np.random.randint(self.mocap_data_len)
        self.start_ind = ind
        # if self.set_mode == "jumpReset" or self.set_mode == "jumpMimicReset":
        qpos = self.motion.data_config[ind].copy()
        qvel = self.motion.data_vel[ind].copy()
        # elif self.set_mode == "jumpResetInit_Mod" or self.set_mode == "jumpResetInit_ModMimic":
        #     qpos = self.init_qpos
        #     qvel = self.init_qvel
        self.set_state(qpos, qvel)
        self.bquat = self.get_body_quat()
        self.cur_t = 0
        self.idx_curr = 0
        self.max_distance = 0.0
        self.r_height = 0.0
        self.imitation_r = 0.0
        self.rewards = [0, 0, 0, 0, 0, 0]
        return np.array(self._get_obs())
    
if __name__ == "__main__":
    curr_path = os.getcwd()
    modelPath = curr_path + "/assets/xml/humanoid_jump.xml"
    filePath = curr_path + "/motions/humanoid3d_jump.txt"
    
    env = HumanoidEnv(modelPath, filePath, render_mode="human")
    obs = env._get_obs()
    action = np.zeros(env.action_dim)
    right_knee_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")
    left_knee_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")
    
    actuator_joint_ids = [env.model.actuator_trnid[i, 0] for i in range(env.model.nu)]
    right_knee_actuator_idx = actuator_joint_ids.index(right_knee_id)
    left_knee_actuator_idx = actuator_joint_ids.index(left_knee_id)
    
    action[right_knee_actuator_idx] = 0.0
    action[left_knee_actuator_idx] = 0.0
    
    import time as t
    for _ in range(10000):
        env.fixed_start = True  # Set fixed start to True for consistent behavior
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            env.reset_model()
        t.sleep(1.0 / env.metadata["render_fps"])
    
    env.close()
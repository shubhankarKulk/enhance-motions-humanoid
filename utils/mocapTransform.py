#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import time
import mujoco
import numpy as np
from os import getcwd
from pyquaternion import Quaternion
from utils.mocapUtils import *
from utils.transformUtils import *
from utils.envUtils import *

#Taken From: https://github.com/mingfeisun/DeepMimic_mujoco

class MocapDM(object):
    def __init__(self):
        self.num_bodies = len(BODY_DEFS)
        self.pos_dim = 3
        self.rot_dim = 4

        curr_path = getcwd()
        xmlpath = '/assets/xml/humanoid_jump_toe.xml'
        with open(curr_path + xmlpath) as fin:
            MODEL_XML = fin.read()

        self.model = load_model_from_xml(MODEL_XML)
        self.sim = MjSim(self.model)

    def load_mocap(self, filepath):
        self.read_raw_data(filepath)
        self.convert_raw_data()

    def load_expert(self):
        with open("motions/expertQPos.json", "r") as f:
            qPos = json.load(f)
        
        # qPos = [v[0] for _, v in sorted(qPos.items(), key=lambda item: int(item[0]))]
        return qPos
    
    def load_toe_expert(self):
        with open("motions/expertQPos_Toe.json", "r") as f:
            qpos = json.load(f)

        return qpos

    def getExpert(self, env, qpos, mode):
        if mode == "noToes":
            expert_qpos = self.load_expert()
        else:
            expert_qpos = self.load_toe_expert()
        expert = {'qpos': expert_qpos}
        feat_keys = {'qvel', 'rlinv', 'rlinv_local', 'rangv', 'rq_rmh',
                 'com', 'head_pos', 'ee_pos', 'ee_wpos', 'bquat', 'bangvel'}
        for key in feat_keys:
            expert[key] = []

        for i in range(np.array(expert_qpos).shape[0]):
            qpos = np.array(expert_qpos[i])
            env.data.qpos[:] = qpos
            mujoco.mj_forward(env.model, env.data)
            
            rq_rmh = deHeading(qpos[3:7])
            ee_pos = env.get_ee_pos()
            bquat = env.get_body_quat()
            com = qpos[:3]
            head_pos = env.get_body_com('neck').copy()
            
            if i > 0:
                prev_qpos = np.array(expert_qpos[i - 1])
                qvel = get_qvel(prev_qpos, qpos, env.dt)
                qvel = qvel.clip(-10.0, 10.0)
                rlinv = qvel[:3].copy()
                rlinv_local = get_quaternion_heading(qvel[:3].copy(), qpos[3:7])
                rangv = qvel[3:6].copy()
                expert['qvel'].append(qvel)
                expert['rlinv'].append(rlinv)
                expert['rlinv_local'].append(rlinv_local)
                expert['rangv'].append(rangv)

            expert['ee_pos'].append(ee_pos)
            expert['bquat'].append(bquat)
            expert['com'].append(com)
            expert['head_pos'].append(head_pos)
            expert['rq_rmh'].append(rq_rmh)
        expert['qvel'].insert(0, expert['qvel'][0].copy())
        expert['rlinv'].insert(0, expert['rlinv'][0].copy())
        expert['rlinv_local'].insert(0, expert['rlinv_local'][0].copy())
        expert['rangv'].insert(0, expert['rangv'][0].copy())
        for i in range(1, np.array(expert_qpos).shape[0]):
            bangvel = get_angvel(expert['bquat'][i - 1], expert['bquat'][i], env.dt)
            expert['bangvel'].append(bangvel)
        expert['bangvel'].insert(0, expert['bangvel'][0].copy())

        return expert
    
    def read_raw_data(self, filepath):
        motions = None
        all_states = []

        durations = []

        with open(filepath, 'r') as fin:
            data = json.load(fin)
            motions = np.array(data["Frames"])
            m_shape = np.shape(motions)
            self.dataset = np.full(m_shape, np.nan)

            total_time = 0.0
            self.dt = motions[0][0]
            for each_frame in motions:
                duration = each_frame[0]
                each_frame[0] = total_time
                total_time += duration
                durations.append(duration)

                curr_idx = 1
                offset_idx = 8
                state = {}
                state['root_pos'] = align_position(each_frame[curr_idx:curr_idx+3])
                # state['root_pos'][2] += 0.08
                state['root_rot'] = align_rotation(each_frame[curr_idx+3:offset_idx])
                for each_joint in BODY_JOINTS_IN_DP_ORDER:
                    curr_idx = offset_idx
                    dof = DOF_DEF[each_joint]
                    if dof == 1:
                        offset_idx += 1
                        state[each_joint] = each_frame[curr_idx:offset_idx]
                    elif dof == 3:
                        offset_idx += 4
                        state[each_joint] = align_rotation(each_frame[curr_idx:offset_idx])
                all_states.append(state)

        self.all_states = all_states
        self.durations = durations

    def calc_rot_vel(self, seg_0, seg_1, dura):
        q_0 = Quaternion(seg_0[0], seg_0[1], seg_0[2], seg_0[3])
        q_1 = Quaternion(seg_1[0], seg_1[1], seg_1[2], seg_1[3])

        q_diff =  q_0.conjugate * q_1
        # q_diff =  q_1 * q_0.conjugate
        axis = q_diff.axis
        angle = q_diff.angle
        
        tmp_diff = angle/dura * axis
        diff_angular = [tmp_diff[0], tmp_diff[1], tmp_diff[2]]

        return diff_angular

    def convert_raw_data(self):
        self.data_vel = []
        self.data_config = []
        self.phase_vals = []
        
        for k in range(len(self.all_states)):
            tmp_vel = []
            tmp_angle = []
            state = self.all_states[k]
            if k == 0:
                dura = self.durations[k]
            else:
                dura = self.durations[k-1]

            # time duration
            init_idx = 0
            offset_idx = 1
            self.dataset[k, init_idx:offset_idx] = dura

            # root pos
            init_idx = offset_idx
            offset_idx += 3
            self.dataset[k, init_idx:offset_idx] = np.array(state['root_pos'])
            if k == 0:
                tmp_vel += [0.0, 0.0, 0.0]
            else:
                tmp_vel += ((self.dataset[k, init_idx:offset_idx] - self.dataset[k-1, init_idx:offset_idx])*1.0/dura).tolist()
            tmp_angle += state['root_pos'].tolist()

            # root rot
            init_idx = offset_idx
            offset_idx += 4
            self.dataset[k, init_idx:offset_idx] = np.array(state['root_rot'])
            if k == 0:
                tmp_vel += [0.0, 0.0, 0.0]
            else:
                tmp_vel += self.calc_rot_vel(self.dataset[k, init_idx:offset_idx], self.dataset[k-1, init_idx:offset_idx], dura)
            tmp_angle += state['root_rot'].tolist()

            for each_joint in BODY_JOINTS:
                init_idx = offset_idx
                tmp_val = state[each_joint]
                if DOF_DEF[each_joint] == 1:
                    assert 1 == len(tmp_val)
                    offset_idx += 1
                    self.dataset[k, init_idx:offset_idx] = state[each_joint]
                    if k == 0:
                        tmp_vel += [0.0]
                    else:
                        tmp_vel += ((self.dataset[k, init_idx:offset_idx] - self.dataset[k-1, init_idx:offset_idx])*1.0/dura).tolist()
                    tmp_angle += state[each_joint].tolist()
                elif DOF_DEF[each_joint] == 3:
                    assert 4 == len(tmp_val)
                    offset_idx += 4
                    self.dataset[k, init_idx:offset_idx] = state[each_joint]
                    if k == 0:
                        tmp_vel += [0.0, 0.0, 0.0]
                    else:
                        tmp_vel += self.calc_rot_vel(self.dataset[k, init_idx:offset_idx], self.dataset[k-1, init_idx:offset_idx], dura)
                    quat = state[each_joint]
                    quat = np.array([quat[1], quat[2], quat[3], quat[0]])
                    euler_tuple = euler_from_quaternion(quat, axes='rxyz')
                    tmp_angle += list(euler_tuple)
                    ## For testing
                    # quat_after = quaternion_from_euler(euler_tuple[0], euler_tuple[1], euler_tuple[2], axes='rxyz')
                    # np.set_printoptions(precision=4, suppress=True)
                    # diff = quat-quat_after
                    # if diff[3] > 0.5:
                    #     import pdb
                    #     pdb.set_trace()
                    #     print(diff)
                phase = (k / len(self.all_states)) * 2 * np.pi
                sin_phase = np.sin(phase)
                cos_phase = np.cos(phase)
                
            self.data_vel.append(np.array(tmp_vel))
            self.data_config.append(np.array(tmp_angle))
            self.phase_vals.append(np.array([sin_phase, cos_phase]))

    def vector_to_rotation_matrix(self, vec):
        """Align z-axis with vector."""
        vec = np.array(vec)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.eye(3)
        vec = vec / norm
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, vec)
        angle = np.arccos(np.clip(np.dot(z_axis, vec), -1.0, 1.0))
        if np.linalg.norm(axis) < 1e-6:
            return np.eye(3)
        axis = axis / np.linalg.norm(axis)
        c, s = np.cos(angle), np.sin(angle)
        t = 1 - c
        x, y, z = axis
        return np.array([
            [t*x*x + c, t*x*y - z*s, t*x*z + y*s],
            [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
            [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
        ])

    def visualize_expert(self, mocap_filepath="/motions/humanoid3d_jump.txt", model_xml_path='/assets/xml/humanoid_jump.xml'):
        from environment.humanoidEnv import HumanoidEnv  # Lazy import
        # Initialize environment
        curr_path = os.getcwd()
        model_path = curr_path + model_xml_path
        file_path = curr_path + mocap_filepath
        env = HumanoidEnv(model_path, file_path, render_mode=None)

        # Load model and create simulation
        with open(model_path) as fin:
            MODEL_XML = fin.read()
        model = load_model_from_xml(MODEL_XML)
        sim = MjSim(model)
        viewer = MjViewer(sim)
        
        # Load expert data
        expert = self.getExpert(env, env.motion.data_config, mode="noToes")
        qpos_list = expert['qpos']
        qvel_list = expert['qvel']
        ee_pos_list = expert['ee_pos']

        env.set_mode = "jump"
        # Replay expert motion with state visualization
        phase_offset = np.array([0.0, 0.0, 0.0])
        ee_names = ["right_ankle", "left_ankle", "right_wrist", "left_wrist"]
        key_bodies = ["torso", "head", "right_upper_arm", "left_upper_arm"]  # Limit bquat visualization
        while True:
            for i in range(len(qpos_list)):
                # Set expert state
                sim_state = sim.get_state()
                sim_state.qpos[:] = qpos_list[i]
                sim_state.qvel[:] = qvel_list[i]
                sim_state.qpos[:3] += phase_offset
                sim.set_state(sim_state)
                sim.forward()

                # Step environment to get current states
                env.reset_model()  # Reset to align with expert
                env.set_state(np.array(qpos_list[i]), np.array(qvel_list[i]))
                env.idx_curr = i % env.mocap_data_len  # Ensure phase advances
                obs = env._get_obs()
                cur_root_pos = env.get_com()  # Root position
                cur_ee = env.get_ee_pos()  # End-effector positions
                cur_ee = np.array(cur_ee).reshape(4, 3)
                cur_phase = env.get_phaseEval()  # Phase
                reward = env.reward()   # Compute reward for display
                # viewer.add_marker(
                #     pos=qpos_list[i][:3],
                #     size=[0.06, 0.06, 0.06],
                #     rgba=[0, 1, 0, 0.8],
                #     type=mujoco.mjtGeom.mjGEOM_SPHERE.value,
                #     label="Expert root_pos"
                # )
                # viewer.add_marker(
                #     pos=cur_root_pos,
                #     size=[0.06, 0.06, 0.06],
                #     rgba=[0, 1, 1, 0.8],
                #     type=mujoco.mjtGeom.mjGEOM_SPHERE.value,
                #     label="Current root_pos"
                # )

                # # 2. End-effector positions (red for expert, blue for current)
                # ee_pos = np.array(ee_pos_list[i]).reshape(-1, 3)
                # cur_ee = np.array(cur_ee).reshape(-1, 3)
                # for j, name in enumerate(ee_names):
                #     viewer.add_marker(
                #         pos=ee_pos[j],
                #         size=[0.05, 0.05, 0.05],
                #         rgba=[1, 0, 0, 0.8],
                #         type=mujoco.mjtGeom.mjGEOM_SPHERE.value,
                #         label=f"Expert {name}"
                #     )
                #     viewer.add_marker(
                #         pos=list(cur_ee[j]),
                #         size=[0.05, 0.05, 0.05],
                #         rgba=[0, 0, 1, 0.8],
                #         type=mujoco.mjtGeom.mjGEOM_SPHERE.value,
                #         label=f"Current {name}"
                #     )

                # viewer.add_overlay(
                #     mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                #     "",
                #     # f"Frame: {i}/{len(qpos_list)} | Reward: {reward:.3f} | Phase: {cur_phase:.3f} | Heading: {cur_heading:.3f} (Expert: {expert_heading:.3f})"
                #     f"Frame: {i}/{len(qpos_list)} | Reward: {reward:.3f} | Phase: {cur_phase:.3f}"
                # )

                # Render the frame
                viewer.render()
                time.sleep(env.dt)

    def play(self, mocap_filepath):
        # Load model and create simulation
        curr_path = getcwd()
        xmlpath = '/assets/xml/humanoid_jump.xml'
        with open(curr_path + xmlpath) as fin:
            MODEL_XML = fin.read()

        model = load_model_from_xml(MODEL_XML)
        sim = MjSim(model)
        viewer = MjViewer(sim)

        qpos = []

        self.read_raw_data(mocap_filepath)
        self.convert_raw_data()

        phase_offset = np.array([0.0, 0.0, 0.0])
        save_config = 0

        # with open("motions/humanoid3d_jump_with_toes.txt", "r") as f:
        #     qpos = json.load(f)
        
        # with open("motions/humanoid3d_jump_with_toes_vel.txt", "r") as g:
        #     qvel = json.load(g)

        # self.data_config = qpos
        # self.data_vel = qvel

        while True:
            for k in range(len(self.data_config)):
                tmp_val = self.data_config[k]
                sim_state = sim.get_state()
                sim_state.qpos[:] = tmp_val[:]
                sim_state.qpos[:3] +=  phase_offset[:]
                sim.set_state(sim_state)
                sim.forward()
                viewer.render()
                qpos.append(tmp_val[:])
            # sim_state = sim.get_state()
            # phase_offset = sim_state.qpos[:3]
            # phase_offset[2] = 0
            if save_config < 1:
                with open("expertQPos_Toe.json", "w") as f:
                    json.dump(convert_ndarray_to_list(qpos), f)
                    save_config += 1
            
def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    else:
        return obj


if __name__ == "__main__":
    test = MocapDM()
    # curr_path = getcwd()
    # test.play(curr_path + "/motions/humanoid3d_jump.txt")
    test.visualize_expert()
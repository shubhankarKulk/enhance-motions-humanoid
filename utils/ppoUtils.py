import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import os
from scipy.interpolate import interp1d
import pickle
import mujoco

class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, hidden_dim, log_std=-2.3):
        super(GaussianActor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.mu_head.weight.data.mul_(0.1)
        self.mu_head.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std, requires_grad=True)

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        return mu

    def get_dist(self,state):
        mu = self.forward(state)
        action_log_std = self.action_log_std.expand_as(mu)
        action_std = torch.exp(action_log_std)

        dist = Normal(mu, action_std)
        return dist

    def deterministic_act(self, state):
        return self.forward(state)

class Critic(nn.Module):
	def __init__(self, state_dim, net_width, hidden_dim):
		super(Critic, self).__init__()

		self.C1 = nn.Linear(state_dim, net_width)
		self.C2 = nn.Linear(net_width, hidden_dim)
		self.C3 = nn.Linear(hidden_dim, 1)

	def forward(self, state):
		v = torch.relu(self.C1(state))
		v = torch.relu(self.C2(v))
		v = self.C3(v)
		return v

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def save_variables(self):
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count
        }

    def load_variables(self, saved_dict):
        self.mean = saved_dict['mean']
        self.var = saved_dict['var']
        self.count = saved_dict['count']

class Normalizer:
    def __init__(self, shape):
        self.rms = RunningMeanStd(shape=shape)

    def __call__(self, obs, update=True):
        if update:
            self.rms.update(obs)
        return (obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)


def logger(writer, losses, total_steps, env):
    ind = env.get_expert_index(env.cur_t)
    writer.add_scalar('actor/loss', losses.get('actor_loss', 0), total_steps)
    writer.add_scalar('critic/loss', losses.get('critic_loss', 0), total_steps)
    writer.add_scalar('actor/advantage_mean', losses.get('adv_mean', 0), total_steps)
    writer.add_scalar('actor/advantage_std', losses.get('adv_std', 0), total_steps)
    writer.add_scalar("CoM/Current_Z", env.data.qpos[2], total_steps)
    writer.add_scalar("CoM/Expert_Z", env.get_expert_attr('qpos', ind)[2], total_steps)
    writer.add_scalar("CoM/Height_Error", abs(env.data.qpos[2] - env.get_expert_attr('qpos', ind)[2]), total_steps)
    writer.add_scalar('Error/position', env.pos_err, total_steps)
    writer.add_scalar('Error/velocity', env.vel_err, total_steps)
    writer.add_scalar('Error/max_torque', env.max_torque, total_steps)

def rewardLogger(writer, avg_score, total_steps, env):
    print(f'Steps: {total_steps}, Avg Reward: {avg_score:.3f}')
    writer.add_scalar("Reward/Imitation", env.rewards[0], total_steps)
    writer.add_scalar("Reward/Explosive", env.rewards[1], total_steps)
    writer.add_scalar("Reward/Distance", env.rewards[2], total_steps)
    writer.add_scalar("Reward/Stability", env.rewards[3], total_steps)
    writer.add_scalar("Reward/Upright", env.rewards[4], total_steps)
    # writer.add_scalar("Reward/CoM", env.rewards[5], total_steps)
    # writer.add_scalar("Reward/Stability", env.rewards[4], total_steps)
    # writer.add_scalar("Reward/Distance", env.rewards[5], total_steps)
    # writer.add_scalar("Reward/MaxDistance", env.max_distance, total_steps)


def jointLogger(writer, total_steps, action, env):
    writer.add_scalar("Torque/Max", env.max_torque, total_steps)
    # Log knee and hip torques
    right_knee_idx = env.actuator_ids[env.joint_names.index("right_knee")]
    left_knee_idx = env.actuator_ids[env.joint_names.index("left_knee")]
    right_hip_y_idx = env.actuator_ids[env.joint_names.index("right_hip_y")]
    left_hip_y_idx = env.actuator_ids[env.joint_names.index("left_hip_y")]
    writer.add_scalar("Torque/Right_Knee", action[right_knee_idx] * env.model.actuator_gear[right_knee_idx, 0], total_steps)
    writer.add_scalar("Torque/Left_Knee", action[left_knee_idx] * env.model.actuator_gear[left_knee_idx, 0], total_steps)
    writer.add_scalar("Torque/Right_Hip_Y", action[right_hip_y_idx] * env.model.actuator_gear[right_hip_y_idx, 0], total_steps)
    writer.add_scalar("Torque/Left_Hip_Y", action[left_hip_y_idx] * env.model.actuator_gear[left_hip_y_idx, 0], total_steps)

    # Log knee and hip joint angles and velocities
    writer.add_scalar("Joint_Angle/Right_Knee", env.data.qpos[7 + env.joint_names.index("right_knee")], total_steps)
    writer.add_scalar("Joint_Angle/Left_Knee", env.data.qpos[7 + env.joint_names.index("left_knee")], total_steps)
    writer.add_scalar("Joint_Angle/Right_Hip_Y", env.data.qpos[7 + env.joint_names.index("right_hip_y")], total_steps)
    writer.add_scalar("Joint_Angle/Left_Hip_Y", env.data.qpos[7 + env.joint_names.index("left_hip_y")], total_steps)
    writer.add_scalar("Joint_Velocity/Right_Knee", env.data.qvel[6 + env.joint_names.index("right_knee")], total_steps)
    writer.add_scalar("Joint_Velocity/Left_Knee", env.data.qvel[6 + env.joint_names.index("left_knee")], total_steps)
    writer.add_scalar("Joint_Velocity/Right_Hip_Y", env.data.qvel[6 + env.joint_names.index("right_hip_y")], total_steps)
    writer.add_scalar("Joint_Velocity/Left_Hip_Y", env.data.qvel[6 + env.joint_names.index("left_hip_y")], total_steps)

cwd = os.getcwd()
def Action_adapter(a,max_action):
	#from [0,1] to [-max,max]
	return  2*(a-0.5)*max_action

def evaluate_policy(env, agent, max_action, turns):
    env.fixed_start = True
    total_scores = 0
    score = []
    k = 0
    for j in range(turns):
        s, _ = env.reset()
        done = False
        while not done:
            a, _ = agent.select_action(s, deterministic=True)
            act = Action_adapter(a, max_action)
            s_next, r, dw, tr, info = env.step(act)
            done = (dw or tr)
            total_scores += r
            s = s_next
            k += 1
        score.append(total_scores)
        total_scores = 0
    return np.mean(score)
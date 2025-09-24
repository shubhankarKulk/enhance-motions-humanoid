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

def stableCalc(env):
    import os
    dir = os.getcwd()
    
    if env.get_phase() > 0.7:    
        com_vel = env.data.qvel[:3]
        com_vel_norm = np.linalg.norm(com_vel)

        if com_vel_norm < 0.5:
            env.stableCounter += 1

        if env.stableCounter == 15:
            return True
    return False    

def rewardAccumulator(results, mode="gear"):
    if mode == "gear":
        scale_key = "gear_scales"
        xlabel = "Average Gear Scaling Factor"
        jump_file = cwd+"/tikz/jump_vs_gear_scale.tex"
        jump_line_file = cwd+"/tikz/jump_vs_gear_scale_line.tex"
        land_file = cwd+"/tikz/landing_vs_gear_scale.tex"
        land_bar_file = cwd+"/tikz/landing_vs_gear_scale_bar.tex"
        title_prefix = "Torque Perturbation"
    else:
        scale_key = "mass_scales"
        xlabel = "Average Mass Scaling Factor"
        jump_file = cwd+"/tikz/jump_vs_mass_scale.tex"
        jump_line_file = cwd+"/tikz/jump_vs_mass_scale_line.tex"
        land_file = cwd+"/tikz/landing_vs_mass_scale.tex"
        land_bar_file = cwd+"/tikz/landing_vs_mass_scale_bar.tex"
        title_prefix = "Mass Perturbation"

    import pandas as pd

    scales = results[scale_key]
    jump_distances = results["jump_distances"]
    landing_success = results["landing_success"]

     # Before plotting, save data to CSV
    data_df = pd.DataFrame({
        scale_key: scales,
        "jump_distances": jump_distances,
        "landing_success": landing_success
    })

    csv_dir = os.path.join(cwd, "data")
    os.makedirs(csv_dir, exist_ok=True)
    data_df.to_csv(os.path.join(csv_dir, f"reward_data_{mode}.csv"), index=False)

    # Scatter Plot: Jump Distance vs Scale
    plt.figure(figsize=(10, 4))
    plt.scatter(scales, jump_distances, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel("Jump Distance (m)")
    plt.title(f"{title_prefix} vs Jump Distance (Scatter)")
    plt.grid(True)
    tikzplotlib.save(jump_file)
    plt.savefig(cwd+f"/viewPlots/jump_{mode}_scatter.png")
    plt.close()

    # Line Plot with Error Bars: Jump Distance vs Scale
    bins = np.linspace(0.8, 1.2, num=9)  # 8 bins between 0.8 and 1.2
    bin_indices = np.digitize(scales, bins)

    means = []
    stds = []
    bin_centers = []
    for i in range(1, len(bins)):
        bin_jumps = [jump_distances[j] for j in range(len(scales)) if bin_indices[j] == i]
        if bin_jumps:
            means.append(np.mean(bin_jumps))
            stds.append(np.std(bin_jumps))
            bin_centers.append((bins[i] + bins[i - 1]) / 2)

    plt.figure(figsize=(10, 4))
    plt.errorbar(bin_centers, means, yerr=stds, fmt='-o', capsize=5, color='tab:blue')
    plt.xlabel(xlabel)
    plt.ylabel("Jump Distance (m)")
    plt.title(f"{title_prefix} vs Jump Distance (Mean ± Std)")
    plt.grid(True)
    tikzplotlib.save(jump_line_file)
    plt.savefig(cwd+f"/viewPlots/jump_{mode}_mean.png")
    plt.close()

    # Scatter Plot: Landing Success vs Scale
    plt.figure(figsize=(10, 4))
    plt.scatter(scales, landing_success, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel("Landing Success (1 = stable)")
    plt.title(f"{title_prefix} vs Stable Landings (Scatter)")
    plt.grid(True)
    tikzplotlib.save(land_file)
    plt.savefig(cwd+f"/viewPlots/land_{mode}_scatter.png")
    plt.close()

    # Bar Plot: Frequency of Stable Landings grouped by bins of scale
    stable_counts = []
    total_counts = []
    group_labels = []
    for i in range(1, len(bins)):
        group = [landing_success[j] for j in range(len(scales)) if bin_indices[j] == i]
        if group:
            stable_counts.append(sum(group))       # count of 1's in bin
            total_counts.append(len(group))        # total samples in bin
            group_labels.append(f"{bins[i-1]:.2f}-{bins[i]:.2f}")

    plt.figure(figsize=(10, 4))
    bars = plt.bar(group_labels, stable_counts, color='lightgreen', edgecolor='green')
    plt.xlabel(xlabel)
    plt.ylabel("Count of Stable Landings")
    plt.title(f"{title_prefix} vs Stable Landing Frequency (Bar Plot)")
    plt.ylim(0, max(total_counts) + 1)
    plt.grid(axis='y')

    # Add counts above bars
    for bar, count in zip(bars, stable_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(count),
                 ha='center', va='bottom', fontsize=9)

    tikzplotlib.save(land_bar_file)
    plt.savefig(cwd+f"/viewPlots/land_{mode}_bar.png")
    plt.close()

def plot_biomechanical_analysis(grf_results):
    cwd = os.getcwd()

    grf_takeoff_all = grf_results["grf_takeoff"]
    grf_landing_all = grf_results["grf_landing"]
    joint_velocities_all = grf_results["joint_velocities"]
    key_events_all = grf_results["key_events"]
    
    joints = list(joint_velocities_all[0].keys())
    num_trials = len(grf_takeoff_all)

    ### 1. Normalize GRF durations ###
    def normalize_grf_sequences(seqs, target_len=100):
        """Resample each GRF sequence to have target_len timesteps."""
        normalized = []
        for arr in seqs:
            if len(arr) == 0:
                normalized.append(np.full((target_len, 2), np.nan))
                continue
            old_x = np.linspace(0, 1, len(arr))
            new_x = np.linspace(0, 1, target_len)
            f_vert = interp1d(old_x, arr[:, 0], kind="linear", fill_value="extrapolate")
            f_horz = interp1d(old_x, arr[:, 1], kind="linear", fill_value="extrapolate")
            normalized.append(np.column_stack([f_vert(new_x), f_horz(new_x)]))
        return np.array(normalized)

    grf_takeoff_norm = normalize_grf_sequences(grf_takeoff_all)
    grf_landing_norm = normalize_grf_sequences(grf_landing_all)

    def plot_grf_phase(data, phase_name):
        mean_grf = np.nanmean(data, axis=0)
        std_grf = np.nanstd(data, axis=0)
        timesteps = np.linspace(0, 100, data.shape[1])  # normalized to %
        
        plt.figure(figsize=(10, 4))
        plt.plot(timesteps, mean_grf[:,0], label='Vertical GRF')
        plt.fill_between(timesteps, mean_grf[:,0]-std_grf[:,0], mean_grf[:,0]+std_grf[:,0], alpha=0.3)
        plt.plot(timesteps, mean_grf[:,1], label='Horizontal GRF')
        plt.fill_between(timesteps, mean_grf[:,1]-std_grf[:,1], mean_grf[:,1]+std_grf[:,1], alpha=0.3)
        plt.xlabel('Normalized Phase (%)')
        plt.ylabel('Force (N)')
        plt.title(f'{phase_name} Phase Ground Reaction Forces (GRF)')
        plt.legend()
        plt.grid(True)
        tikzplotlib.save(os.path.join(cwd, f"tikz/{phase_name}_phase.tex"))
        plt.savefig(os.path.join(cwd, f"viewPlots/{phase_name}.png"))

    plot_grf_phase(grf_takeoff_norm, "Takeoff")
    plot_grf_phase(grf_landing_norm, "Landing")

    ### 2. Joint Velocity Profiles (unchanged) ###
    max_len = max(len(joint_velocities_all[i][joints[0]]) for i in range(num_trials))
    joint_vels_padded = {joint: np.full((num_trials, max_len), np.nan) for joint in joints}

    for i, trial in enumerate(joint_velocities_all):
        for joint in joints:
            vals = trial[joint]
            joint_vels_padded[joint][i, :len(vals)] = vals

    plt.figure(figsize=(10,6))
    timesteps = np.arange(max_len)
    for joint in joints:
        mean_vel = np.nanmean(joint_vels_padded[joint], axis=0)
        std_vel = np.nanstd(joint_vels_padded[joint], axis=0)
        plt.plot(timesteps, mean_vel, label=f'{joint} velocity')
        plt.fill_between(timesteps, mean_vel - std_vel, mean_vel + std_vel, alpha=0.3)

    plt.xlabel('Time Step')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Joint Velocity Profiles')
    plt.legend()
    plt.grid(True)
    tikzplotlib.save(os.path.join(cwd, "tikz/joint_vel_profile.tex"))
    plt.savefig(os.path.join(cwd, "viewPlots/profile.png"))

    ### 3. Timing Histograms (unchanged) ###
    takeoffs = [e["takeoff"] for e in key_events_all if e["takeoff"] is not None]
    apexes = [e["apex"] for e in key_events_all if e["apex"] is not None]
    landings = [e["landing"] for e in key_events_all if e["landing"] is not None]

    plt.figure(figsize=(10,4))
    bins = 20
    plt.hist(takeoffs, bins=bins, alpha=0.7, label='Takeoff')
    plt.hist(apexes, bins=bins, alpha=0.7, label='Apex')
    plt.hist(landings, bins=bins, alpha=0.7, label='Landing')
    plt.xlabel('Time Step')
    plt.ylabel('Frequency')
    plt.title('Timing Alignment of Key Jump Events')
    plt.legend()
    plt.grid(True)
    tikzplotlib.save(os.path.join(cwd, "tikz/key_elements.tex"))
    plt.savefig(os.path.join(cwd, "viewPlots/key_elements.png"))

    ### 4. Save raw data (unchanged) ###
    npz_dir = os.path.join(cwd, "data")
    os.makedirs(npz_dir, exist_ok=True)
    grf_takeoff_arr = np.array([np.array(arr) for arr in grf_takeoff_all], dtype=object)
    grf_landing_arr = np.array([np.array(arr) for arr in grf_landing_all], dtype=object)
    joint_velocities_arr = np.array(joint_velocities_all, dtype=object)
    with open(os.path.join(npz_dir, "key_events.pkl"), "wb") as f:
        pickle.dump(key_events_all, f)
    np.savez(os.path.join(npz_dir, "biomechanical_data.npz"),
             grf_takeoff=grf_takeoff_arr,
             grf_landing=grf_landing_arr,
             joint_velocities=joint_velocities_arr
            )
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

def evaluate_policy_metrics(env, agent, max_action, turns):
    distances = []
    imitation_errors = []
    symmetry_scores = []
    energy_efficiencies = []
    env.fixed_start = True
    total_scores = 0
    score = []

    for _ in range(turns):
        s, _ = env.reset()
        done = False
        total_distance = 0
        episode_energy = 0.0
        total_distance = 0.0
        max_com_height = 0.0
        symmetry_score = 0.0
        imitation_error = 0.0

        while not done:
            a, _ = agent.select_action(s, deterministic=True)
            act = Action_adapter(a, max_action)
            s_next, r, dw, tr, info = env.step(act)
            done = (dw or tr)
            total_scores += r
            s = s_next
            
            episode_energy += np.sum(np.square(env.data.ctrl)) * env.dt
            total_distance = info.get("jump_distance", total_distance)
            symmetry_score = info.get("symmetry_score", symmetry_score)
            imitation_error = info.get("imitation_mse", imitation_error)
        
        distances.append(total_distance)
        symmetry_scores.append(symmetry_score)
        imitation_errors.append(imitation_error)

        if episode_energy > 1e-8:
            energy_efficiencies.append(total_distance/episode_energy)
        else:
            energy_efficiencies.append(0.0)

        score.append(total_scores)
        total_scores = 0

    print(f"Mean Distances: {np.mean(distances)}")
    # print(f"Success Rate: {np.mean(success_rates)*100}%")
    print(f"Imitation MSE: {np.mean(imitation_errors)}")
    print(f"Symmmetry Score: {np.mean(symmetry_scores)}")
    # print(f"Mean COM Heights: {np.mean(com_heights)}")
    print(f"Energy Efficiency: {np.mean(energy_efficiencies)}")
    return np.mean(score)

def evaluate_data(env, agent, max_action, turns):
    import mujoco
    import numpy as np

    env.fixed_start = True
    total_scores = 0

    # Results storage
    mass_results = {
        "scores": [],
        "mass_scales": [],
        "jump_distances": [],
        "landing_success": []
    }
    gear_results = {
        "scores": [],
        "gear_scales": [],
        "jump_distances": [],
        "landing_success": []
    }
    grf_results = {
        "scores": [],
        "grf_takeoff": [],       # GRF vectors during takeoff phase
        "grf_landing": [],       # GRF vectors during landing phase
        "joint_velocities": [],  # dictionary with joint velocities over time
        "key_events": []         # timing info for takeoff, apex, landing
    }

    # Define joints of interest for velocity profiles
    joints_of_interest = ["right_hip_y", "left_hip_y", "right_knee", "left_knee", "right_ankle_y", "left_ankle_y"]  # replace with exact joint names in your env

    for mode in ["mass", "gear", "grf"]:
        print(f"Mode {mode}")
        for j in range(turns):
            if mode == "mass":
                mass_scale_factors = []
                for body_name in env.modified_body_masses:
                    scale = np.random.uniform(0.8, 1.2)
                    env.modified_body_masses[body_name] = env.base_body_masses[body_name] * scale
                    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                    env.model.body_mass[body_id] = env.modified_body_masses[body_name]
                    mass_scale_factors.append(scale)
                avg_mass_scale = np.mean(mass_scale_factors)
                mass_results["mass_scales"].append(avg_mass_scale)

            if mode == "gear":
                gear_scale_factors = []
                for i in range(env.model.nu):
                    scale = np.random.uniform(0.8, 1.2)
                    env.modified_actuator_gears[i] = env.base_actuator_gears[i] * scale
                    env.model.actuator_gear[i] = env.modified_actuator_gears[i]
                    gear_scale_factors.append(scale)
                avg_gear_scale = np.mean(gear_scale_factors)
                gear_results["gear_scales"].append(avg_gear_scale)

            # Reset environment and initialize storage for grf/joint velocity/timing
            s, _ = env.reset()
            done = False
            initial_com_x = env.data.qpos[0]

            # For biomechanical data collection in "grf" mode
            grf_takeoff = []
            grf_landing = []
            all_grf_vectors = []
            joint_velocities = {joint: [] for joint in joints_of_interest}
            key_events = {"takeoff": None, "apex": None, "landing": None}
            on_ground = True  # Track contact state for takeoff and landing detection
            timestep = 0
            max_timesteps = 1000  # safety cap

            max_com_height = -np.inf
            apex_timestep = None

            while not done and timestep < max_timesteps:
                a, _ = agent.select_action(s, deterministic=True)
                act = Action_adapter(a, max_action)
                s_next, r, dw, tr, info = env.step(act)

                # Stable landing calculation (you might want to update this for your logic)
                land = stableCalc(env)

                # Record GRF - vertical and horizontal forces (assuming env provides contact forces)
                contact_forces = env.data.cfrc_ext.copy()  # External contact forces on bodies, shape (nbodies, 6)
                foot_ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in ['foot_l', 'foot_r']]
                vertical_force = 0
                horizontal_force = 0
                for fid in foot_ids:
                    f = contact_forces[fid, :3]  # forces in x,y,z directions
                    vertical_force += f[2]  # z axis usually vertical
                    horizontal_force += np.linalg.norm(f[:2])  # magnitude of horizontal (x,y) forces

                phase = env.get_phase() if hasattr(env, "get_phase") else timestep / max_timesteps

                all_grf_vectors.append([vertical_force, horizontal_force])

                if phase < 0.3:
                    grf_takeoff.append([vertical_force, horizontal_force])
                elif phase > 0.7:
                    grf_landing.append([vertical_force, horizontal_force])

                # Record joint velocities (angular velocities)
                for joint_name in joints_of_interest:
                    joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    joint_vel = env.data.qvel[joint_id]
                    joint_velocities[joint_name].append(joint_vel)

                # Foot contact detection
                contact_threshold = 20.0  # Newtons threshold to consider foot contact
                foot_contact = vertical_force > contact_threshold

                # Detect takeoff event
                if key_events["takeoff"] is None and not foot_contact and on_ground and timestep > 40:
                    key_events["takeoff"] = timestep
                    on_ground = False
                    print(f"Takeoff detected at timestep {timestep}")

                # Track apex (max center of mass height)
                com_height = env.data.qpos[2]  # assuming z-axis is height
                if com_height > max_com_height:
                    max_com_height = com_height
                    apex_timestep = timestep

                # Detect landing event
                if key_events["takeoff"] is not None and foot_contact and not on_ground and timestep > 60:
                    key_events["landing"] = timestep
                    on_ground = True
                    print(f"Landing detected at timestep {timestep}")

                done = (dw or tr)
                total_scores += r
                s = s_next
                timestep += 1

            key_events["apex"] = apex_timestep
            print(f"Trial key events: {key_events}")

            final_com_x = env.data.qpos[0]
            jump_distance = final_com_x - initial_com_x

            if mode == "mass":
                mass_results["scores"].append(total_scores)
                mass_results["jump_distances"].append(jump_distance)
                mass_results["landing_success"].append(int(land))
            elif mode == "gear":
                gear_results["scores"].append(total_scores)
                gear_results["jump_distances"].append(jump_distance)
                gear_results["landing_success"].append(int(land))
            else:  # mode == "grf"                
                takeoff_idx = key_events["takeoff"]
                landing_idx = key_events["landing"]

                if takeoff_idx is not None:
                    grf_takeoff = np.array(all_grf_vectors[:takeoff_idx])  # frames before takeoff
                else:
                    grf_takeoff = np.empty((0, 2))  # no takeoff detected

                if landing_idx is not None:
                    grf_landing = np.array(all_grf_vectors[landing_idx:])  # frames after landing
                else:
                    grf_landing = np.empty((0, 2))
                
                grf_results["grf_takeoff"].append(grf_takeoff)
                grf_results["grf_landing"].append(grf_landing)
                grf_results["scores"].append(total_scores)
                grf_results["joint_velocities"].append(joint_velocities)
                grf_results["key_events"].append(key_events)

            total_scores = 0

    rewardAccumulator(mass_results, mode="mass")
    rewardAccumulator(gear_results, mode="gear")
    plot_biomechanical_analysis(grf_results)

    return np.mean(mass_results["scores"] + gear_results["scores"] + grf_results["scores"])

def evaluate_rest(env, agent, max_action, turns):

    env.fixed_start = True
    results = {"episodes": []}
    dt = env.model.opt.timestep  # base sim step; adjust if you step multiple times per action

    joints_of_interest = ["right_hip_y", "left_hip_y",
                          "right_knee", "left_knee",
                          "right_ankle_y", "left_ankle_y"]

    # Precompute qpos/qvel indices for the joints (assumes hinge joints)
    joint_indices = {}
    for name in joints_of_interest:
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_idx = env.model.jnt_qposadr[jid]
        qvel_idx = env.model.jnt_dofadr[jid]
        joint_indices[name] = (int(qpos_idx), int(qvel_idx))

    scores = []
    for _ in range(turns):
        ep = {
            "t": [],
            "dt": dt,
            "phase": [],
            "agent": {
                "com": [],          # shape (T, 3)
                "ee_pos": [],       # shape (T, K, 3) – ensure your getter returns fixed K
                "qpos": {j: [] for j in joints_of_interest},
                "qvel": {j: [] for j in joints_of_interest},
            },
            "expert": {
                "com": [],
                "ee_pos": [],
                "qpos": {j: [] for j in joints_of_interest},
                "qvel": {j: [] for j in joints_of_interest},
            }
        }

        s, _ = env.reset()
        done = False
        ep_return = 0.0
        episode_energy = 0
        total_distance = 0
        energy_efficiencies = []

        while not done:
            t = getattr(env, "cur_t", len(ep["t"]))
            # expert frame index
            ind = env.get_expert_index(t)
            ind = min(ind, env.mocap_data_len - 1)

            # act
            a, _ = agent.select_action(s, deterministic=True)
            act = Action_adapter(a, max_action)
            s_next, r, dw, tr, info = env.step(act)
            done = (dw or tr)
            ep_return += r

            episode_energy += np.sum(np.square(env.data.ctrl)) * env.dt
            total_distance = info.get("jump_distance", total_distance)

            # time & phase
            ep["t"].append(t)
            phase = env.get_phase() if hasattr(env, "get_phase") else None
            ep["phase"].append(phase)

            # COM & EE (ensure getters return consistent shapes)
            agent_com = np.asarray(env.get_com(), dtype=float)        # (3,)
            agent_ee = np.asarray(env.get_ee_pos(), dtype=float)      # (K, 3)
            exp_com   = np.asarray(env.get_expert_attr('com',    ind), dtype=float)  # (3,)
            exp_ee    = np.asarray(env.get_expert_attr('ee_pos', ind), dtype=float)  # (K, 3)

            ep["agent"]["com"].append(agent_com)
            ep["agent"]["ee_pos"].append(agent_ee)
            ep["expert"]["com"].append(exp_com)
            ep["expert"]["ee_pos"].append(exp_ee)

            # joints
            for jn, (qp_i, qv_i) in joint_indices.items():
                # agent
                ep["agent"]["qpos"][jn].append(float(env.data.qpos[qp_i]))
                ep["agent"]["qvel"][jn].append(float(env.data.qvel[qv_i]))
                # expert
                exp_qpos = env.get_expert_attr('qpos', ind)
                exp_qvel = env.get_expert_attr('qvel', ind)
                ep["expert"]["qpos"][jn].append(float(exp_qpos[qp_i]))
                ep["expert"]["qvel"][jn].append(float(exp_qvel[qv_i]))

            s = s_next

        # convert lists to arrays
        ep["t"]      = np.asarray(ep["t"], dtype=int)
        ep["phase"]  = np.asarray(ep["phase"]) if any(p is not None for p in ep["phase"]) else None
        ep["agent"]["com"]   = np.vstack(ep["agent"]["com"])
        ep["expert"]["com"]  = np.vstack(ep["expert"]["com"])
        ep["agent"]["ee_pos"]  = np.stack(ep["agent"]["ee_pos"], axis=0)
        ep["expert"]["ee_pos"] = np.stack(ep["expert"]["ee_pos"], axis=0)
        for jn in joints_of_interest:
            ep["agent"]["qpos"][jn] = np.asarray(ep["agent"]["qpos"][jn], dtype=float)
            ep["agent"]["qvel"][jn] = np.asarray(ep["agent"]["qvel"][jn], dtype=float)
            ep["expert"]["qpos"][jn] = np.asarray(ep["expert"]["qpos"][jn], dtype=float)
            ep["expert"]["qvel"][jn] = np.asarray(ep["expert"]["qvel"][jn], dtype=float)

        if episode_energy > 1e-8:
            energy_efficiencies.append(total_distance/episode_energy)
        else:
            energy_efficiencies.append(0.0)

        results["episodes"].append(ep)
        scores.append(ep_return)

    # remaining_plots(ep, joints_of_interest, energy_efficiencies)
    np.savez(os.path.join(cwd, "data", "energy_eff__agent.npz"), energy_efficiencies)
    np.savez(os.path.join(cwd, "data", "remaining_data.npz"), ep)

    results["mean_return"] = float(np.mean(scores))
    return results["mean_return"]

def evaluate_foot_prog(env, agent, max_action, turns=5):
    """
    Evaluate foot landing performance of an agent vs expert.

    Returns:
        mean_score: Average reward over episodes
    """
    env.fixed_start = True
    scores = []

    # Observation dictionary in the format expected by plotting functions
    observe = {
        "agent": {"foot_R": [], "foot_L": [], "com": []},
        "expert": {"foot_R": [], "foot_L": [], "com": []},
        "dt": env.model.opt.timestep  # optional time step
    }

    # Map body names to foot labels
    body_map = {"right_ankle": "foot_R", "left_ankle": "foot_L"}

    # Precompute body IDs
    body_ids = {name: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in body_map.keys()}

    for ep in range(turns):
        s, _ = env.reset()
        done = False
        total_reward = 0

        # Temporary lists for this episode
        ep_agent = {k: [] for k in observe["agent"].keys()}
        ep_expert = {k: [] for k in observe["expert"].keys()}

        while not done:
            t = getattr(env, "cur_t", 0)
            ind = min(env.get_expert_index(t), env.mocap_data_len - 1)

            # Agent action
            a, _ = agent.select_action(s, deterministic=True)
            act = Action_adapter(a, max_action)
            s_next, r, dw, tr, info = env.step(act)
            done = dw or tr
            total_reward += r
            s = s_next

            # Record foot body positions (x,y,z) and CoM
            for body_name, label in body_map.items():
                bid = body_ids[body_name]
                ep_agent[label].append(env.data.xpos[bid].copy())
                # ep_expert[label].append(env.get_expert_attr("xpos", ind, bid))

            ep_agent["com"].append(env.data.subtree_com[0].copy())     # root CoM
            ep_expert["com"].append(env.get_expert_attr("com", ind))

        # Convert episode lists to arrays and append to global observe
        for key in ep_agent.keys():
            observe["agent"][key].append(np.array(ep_agent[key]))
            # observe["expert"][key].append(np.array(ep_expert[key]))

        scores.append(total_reward)

    # Plot landing analysis (use last episode or average across episodes)
    # plot_landing_analysis(observe)

    # Save the entire observation dictionary
    np.savez(os.path.join(cwd, "data", "foot_progression.npz"), **observe)

    return np.mean(scores)

def evaluate_land_prog(env, agent, max_action, turns=5):
    """
    Evaluate foot landing performance of an agent vs expert.

    Returns:
        mean_score: Average reward over episodes
    """
    env.fixed_start = True
    scores = []

    # Observation dictionary in the format expected by plotting functions
    observe = {
        "agent": {"foot_R": [], "foot_L": [], "com": []},
        "expert": {"foot_R": [], "foot_L": [], "com": []},
        "dt": env.model.opt.timestep  # optional time step
    }

    # Map body names to foot labels
    body_map = {"right_ankle": "foot_R", "left_ankle": "foot_L"}

    # Precompute body IDs
    body_ids = {name: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in body_map.keys()}

    for ep in range(turns):
        s, _ = env.reset()
        done = False
        total_reward = 0

        # Temporary lists for this episode
        ep_agent = {k: [] for k in observe["agent"].keys()}
        ep_expert = {k: [] for k in observe["expert"].keys()}

        while not done:
            t = getattr(env, "cur_t", 0)
            ind = min(env.get_expert_index(t), env.mocap_data_len - 1)

            # Agent action
            a, _ = agent.select_action(s, deterministic=True)
            act = Action_adapter(a, max_action)
            s_next, r, dw, tr, info = env.step(act)
            done = dw or tr
            total_reward += r
            s = s_next

            # Record foot body positions (x,y,z) and CoM
            for body_name, label in body_map.items():
                bid = body_ids[body_name]
                ep_agent[label].append(env.data.xpos[bid].copy())
                # ep_expert[label].append(env.get_expert_attr("xpos", ind, bid))

            ep_agent["com"].append(env.data.subtree_com[0].copy())     # root CoM
            ep_expert["com"].append(env.get_expert_attr("com", ind))

        # Convert episode lists to arrays and append to global observe
        for key in ep_agent.keys():
            observe["agent"][key].append(np.array(ep_agent[key]))
            # observe["expert"][key].append(np.array(ep_expert[key]))

        scores.append(total_reward)

    # Plot landing analysis (use last episode or average across episodes)
    # plot_landing_analysis(observe)

    # Save the entire observation dictionary
    np.savez(os.path.join(cwd, "data", "foot_progression.npz"), **observe)

    return np.mean(scores)
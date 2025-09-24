import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np
from numpy import load
import os
from dtw import accelerated_dtw
import string

cwd = os.getcwd()
TOUCHDOWN_Z = 0.1

def main():
    joints_of_interest = ["left_hip_y", "right_hip_y",
                          "left_knee", "right_knee",
                          "left_ankle_y", "right_ankle_y"]
    
    labels = ["Left Foot", "Right Foot", "Left Hand", "Right Hand"]
    
    data = load(cwd + '/data/remaining_data.npz', allow_pickle=True)
    ep = data["arr_0"].item() 
    # plot_joint_tracking(ep, joints_of_interest)
    # plot_vel_tracking(ep, joints_of_interest)
    # plot_ee_trajectory(ep, labels)
    # plot_smoothness(ep)
    # plot_dtw(ep)
    plot_com(ep)
    # plot_mse_per_joint(ep, joints_of_interest)
    
    data = load(cwd + '/data/foot_progression.npz', allow_pickle=True)
    expert = load(cwd + '/data/expert_foot.npz', allow_pickle=True)
    ep = {
        "agent": data["agent"].item(),
        "expert": expert["expert"].item(),
        "dt": data["dt"].item() if data["dt"].ndim == 0 else data["dt"],
    }
    # plot_touchdown_locations(ep)
    # plot_symmetry_error(ep)

def plot_joint_tracking(ep, joints_of_interest):
    t = ep["t"] * ep["dt"]  # convert to seconds
    for i, joint in enumerate(joints_of_interest):
        plt.plot(t, ep["agent"]["qpos"][joint], label="Agent")
        plt.plot(t, ep["expert"]["qpos"][joint], label="Expert")
        plt.xlabel("Time [s]")
        plt.ylabel(f"Angle [rad]")
        plt.title(f"{string.capwords(joint.split('_')[0])} {string.capwords(joint.split('_')[1])}")
        plt.legend(loc="upper right", fontsize=9, frameon=True)
        
        plt.savefig(cwd+f"/viewPlots/{joint.split('_')[0]}_{joint.split('_')[1]}_tracking.png")
        tikzplotlib.save(os.path.join(cwd, f"tikz/{joint.split('_')[0]}_{joint.split('_')[1]}_tracking.tex"))
        plt.close()

def plot_vel_tracking(ep, joints_of_interest):
    t = ep["t"] * ep["dt"]  # convert to seconds
    for i, joint in enumerate(joints_of_interest):
        plt.plot(t, ep["agent"]["qvel"][joint], label="Agent")
        plt.plot(t, ep["expert"]["qvel"][joint], label="Expert")
        plt.xlabel("Time [s]")
        plt.ylabel(f"Velocity [m/s]")
        plt.title(f"{string.capwords(joint.split('_')[1])}")
        plt.legend(loc="upper right", fontsize=9, frameon=True)      
        plt.savefig(cwd+f"/viewPlots/{joint.split('_')[0]}_{joint.split('_')[1]}_vel_tracking.png")
        tikzplotlib.save(os.path.join(cwd, f"tikz/{joint.split('_')[0]}_{joint.split('_')[1]}_vel_tracking.tex"))
        plt.close()
    
import matplotlib.lines as mlines

def plot_com(ep):
    t = ep["t"] * ep["dt"]
    plt.figure()
    colors = {"x":"C0", "y":"C1", "z":"C2"}
    
    # plot
    for i, label in enumerate(["x","y","z"]):
        plt.plot(t, ep["agent"]["com"][:,i], color=colors[label], linestyle="-")
        plt.plot(t, ep["expert"]["com"][:,i], color=colors[label], linestyle="--")
    
    plt.xlabel("Time [s]")
    plt.ylabel("COM position [m]")
    plt.title("Center of Mass trajectory")

    # legend 1: agent vs expert
    agent_line = mlines.Line2D([], [], color="black", linestyle="-", label="Agent")
    expert_line = mlines.Line2D([], [], color="black", linestyle="--", label="Expert")
    legend1 = plt.legend(handles=[agent_line, expert_line],
                         loc="upper left", frameon=True)
    plt.gca().add_artist(legend1)  # keep this legend when adding the next one

    # legend 2: x/y/z color coding
    color_handles = [mlines.Line2D([], [], color=colors[k], linestyle="-", label=k.upper())
                     for k in ["x","y","z"]]
    plt.legend(handles=color_handles, loc="upper center", ncol=3, frameon=True)

    plt.tight_layout()
    plt.savefig(cwd+f"/viewPlots/com_tracking.png")
    tikzplotlib.save(os.path.join(cwd, f"tikz/com_tracking.tex"))
    plt.close()

def plot_ee_trajectory(ep, labels, n_eff=4):

    agent = np.array(ep["agent"]["ee_pos"]).reshape(-1, n_eff, 3)
    expert = np.array(ep["expert"]["ee_pos"]).reshape(-1, n_eff, 3)

    for i, label in enumerate(labels):
        agent_i = agent[:, i, :]
        expert_i = expert[:, i, :]

        plt.plot(agent_i[:, 0], agent_i[:, 2], label=f"Agent {label}")   # X-Z plane
        plt.plot(expert_i[:, 0], expert_i[:, 2], label=f"Expert {label}")
        plt.xlabel("Forward [m]")
        plt.ylabel("Height [m]")
        plt.title(f"End-effector trajectory: {label}")
        plt.legend(loc="upper right", fontsize=9, frameon=True)
        plt.savefig(cwd + f"/viewPlots/{label}_tracking.png")
        tikzplotlib.save(os.path.join(cwd, f"tikz/{label}_tracking.tex"))
        plt.close()

def plot_mse_per_joint(ep, joints):
    mse = []
    for jn in joints:
        a = np.array(ep["agent"]["qpos"][jn])
        e = np.array(ep["expert"]["qpos"][jn])
        mse.append(np.mean((a - e)**2))

    plt.bar([f"{string.capwords(jn.split('_')[0])} {string.capwords(jn.split('_')[1])}" for jn in joints], mse)
    plt.ylabel("MSE [rad²]")
    plt.title("Joint angle tracking error")
    # plt.show()
    plt.savefig(cwd+f"/viewPlots/mse_plot.png")
    tikzplotlib.save(os.path.join(cwd, f"tikz/mse_plot.tex"))
    plt.close()

def plot_smoothness(ep):
    dt = ep["dt"]

    joint_pairs = [
        ("left_hip_y", "right_hip_y"),
        ("left_knee", "right_knee"),
        ("left_ankle_y", "right_ankle_y")
    ]
    
    for i, (l, r) in enumerate(joint_pairs):
        qvel_l = ep["agent"]["qvel"][l]
        acc_l  = np.diff(qvel_l)/dt
        qvel_r = ep["agent"]["qvel"][r]
        acc_r  = np.diff(qvel_r)/dt

        plt.plot(acc_l, label=f"{string.capwords(l.split('_')[0])} {string.capwords(l.split('_')[1])} acceleration")
        plt.plot(acc_r, label=f"{string.capwords(r.split('_')[0])} {string.capwords(r.split('_')[1])} acceleration")
        plt.title(f"{l.split('_')[1]}")
        plt.ylabel("Acceleration [rad/sec²]")
        plt.xlabel("Phase")
        plt.legend(loc="upper right", fontsize=9, frameon=True)
        plt.savefig(cwd+f"/viewPlots/{l.split('_')[1]}_acceleration.png")
        tikzplotlib.save(os.path.join(cwd, f"tikz/{l.split('_')[1]}_acceleration.tex"))
        plt.close()

import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
import pandas as pd

def plot_dtw(ep):
    t = ep["t"] * ep["dt"]
    a = np.array(ep["agent"]["com"][:,0])
    e = np.array(ep["expert"]["com"][:,0])

    distance, paths = dtw.warping_paths(a, e, use_c=False)
    best_path = dtw.best_path(paths)
    similarity_score = distance / len(best_path)

    print(similarity_score, distance)
    plt.plot(a, label='Agent CoM', color='blue', marker='o')
    plt.plot(e, label='Expert CoM', color='orange', marker='x', linestyle='--')
    for c, d in best_path:
        plt.plot([c, d], [a[c], e[d]], color='grey', linestyle='-', linewidth=1, alpha = 0.5)
    plt.title('Point-to-Point Comparison After DTW Alignment')
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("COM position [m]")
    plt.savefig(cwd+f"/viewPlots/dtw_plot.png")
    tikzplotlib.save(os.path.join(cwd, f"tikz/dtw_plot.tex"))
    plt.close()
    
def touchdown_index(traj, printVal=False):
    idxs = np.where(traj[:,2] < TOUCHDOWN_Z)[0]
    if printVal:
        pass
        # print(traj)
    return idxs[0] if len(idxs) > 0 else -1

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

def plot_touchdown_locations(ep):
    # Enable LaTeX rendering for all text
    plt.rcParams.update({
        "text.usetex": False,   # don't call external latex
        "mathtext.fontset": "cm",  # Computer Modern
        "font.family": "serif",    # serif family
    })

    foot_names = ["foot_L", "foot_R"]
    colors = {"foot_L": ("red", "blue"), "foot_R": ("orange", "green")}
    
    plt.figure(figsize=(8,6))

    for foot in foot_names:
        expert_color, agent_color = colors[foot]

        # --- Expert (single trajectory) ---
        expert_traj = np.array(ep["expert"][foot])  # 107 x 3
        idx_e = touchdown_index(expert_traj, "expert_traj")
        plt.plot(expert_traj[:,0], expert_traj[:,1], '--', color=expert_color, alpha=0.7, label=fr"Expert {foot} traj")
        if idx_e >= 0:
            plt.scatter(*expert_traj[idx_e,:2], color=expert_color, s=50)

        # --- Agent (multiple trajectories) ---
        agent_trajs = np.array(ep["agent"][foot])  # 150 x 150 x 3
        mean_traj = np.mean(agent_trajs, axis=0)   # 150 x 3
        std_traj  = np.std(agent_trajs, axis=0)    # 150 x 3

        # Touchdown index on mean trajectory
        idx_a = touchdown_index(mean_traj, f"agent_mean_{foot}")

        # Plot mean trajectory
        plt.plot(mean_traj[:,0], mean_traj[:,1], '-', color=agent_color, label=fr"Agent {foot} mean traj")

        # Shaded region for ±1 std
        plt.fill_between(
            mean_traj[:,0],
            mean_traj[:,1] - std_traj[:,1],
            mean_traj[:,1] + std_traj[:,1],
            color=agent_color,
            alpha=0.2)

        # Touchdown marker
        if idx_a >= 0:
            plt.scatter(*mean_traj[idx_a,:2], color=agent_color, s=50)

    plt.xlabel(r"$X$ position[m]")
    plt.ylabel(r"$Y$ position[m]")
    plt.title(r"Foot Trajectories and Touchdowns")
    plt.axis('equal')
    plt.legend(loc="upper right", fontsize=9, frameon=True)
    plt.savefig(cwd+"/viewPlots/foot_touchdowns.png", dpi=300, bbox_inches="tight")
    tikzplotlib.save(cwd+"/tikz/foot_touchdowns.tex")
    plt.close()

def plot_symmetry_error(ep):
    """Bar plot showing lateral symmetry error at landing."""
    foot_names = ["foot_L", "foot_R"]

    # --- Expert (single trajectory) ---
    expert_L = np.array(ep["expert"][foot_names[0]])
    expert_R = np.array(ep["expert"][foot_names[1]])
    idx_Le = touchdown_index(expert_L)
    idx_Re = touchdown_index(expert_R)

    if idx_Le < 0 or idx_Re < 0:
        print("Warning: expert touchdown not detected.")
        return

    sym_expert = abs(expert_L[idx_Le,1] - expert_R[idx_Re,1])

    # --- Agent (multiple trajectories) ---
    agent_L = np.array(ep["agent"][foot_names[0]])  # shape: num_eps x T x 3
    agent_R = np.array(ep["agent"][foot_names[1]])  # shape: num_eps x T x 3

    num_eps = agent_L.shape[0]

    # Compute symmetry for each episode at touchdown, then take mean
    sym_agent_list = []
    for ep_idx in range(num_eps):
        traj_L = agent_L[ep_idx]
        traj_R = agent_R[ep_idx]

        idx_La = touchdown_index(traj_L)
        idx_Ra = touchdown_index(traj_R)

        if idx_La >= 0 and idx_Ra >= 0:
            sym_agent_list.append(abs(traj_L[idx_La,1] - traj_R[idx_Ra,1]))

    if len(sym_agent_list) == 0:
        print("Warning: agent touchdowns not detected.")
        return

    sym_agent = np.mean(sym_agent_list)

    # --- Plot ---
    plt.figure()
    plt.bar(["Expert","Agent"], [sym_expert, sym_agent], color=["gray","blue"])
    plt.ylabel("Lateral Foot Distance [m]")
    plt.title("Foot Symmetry at Landing")
    plt.savefig(cwd+"/viewPlots/foot_symmetry.png")
    tikzplotlib.save(cwd+"/tikz/foot_symmetry.tex")
    plt.close()
    
if __name__ == "__main__":
    main()
# %%

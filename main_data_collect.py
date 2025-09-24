import math
import random
import numpy as np
from omegaconf import OmegaConf
import torch
import os
from mujoco_py import load_model_from_path, MjSim
import time as t
import datetime
import argparse
from environment.humanoidEnvTesting import HumanoidEnv
from algorithm.ppo import PPO_Agent
from utils.ppoUtils import evaluate_policy, Action_adapter, logger, rewardLogger, jointLogger
from utils.ppoUtils import evaluate_policy_metrics
import multiprocessing as mp

trainingModes = [
    "jump",         # Default mode, no ablation or mass/torque changes
    "mass_changes",           # Mass changes
    "torque_changes",         # Torque changes
    "mass_torque",            # Mass and torque changes
    "ablation_no_imitation",  # No imitation reward
    "ablation_no_height",     # No height reward
    "ablation_no_stability",  # No stability reward
    "ablation_no_distance",   # No leg matching reward
    "ablation_no_squat",
    "ablation_no_leg",
    "ablation_no_sym",
    "ablation_no_takeoff",
    "ablation_no_yaw",
    "ablation_no_upright"
]

# Function to set optimizer learning rate
def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Adaptive parameter tuning class
class UpdateConfig:
    def __init__(self, config):
        self.config = config
        # Initialize adaptive parameters
        self.policy_lr = config.a_lr
        self.log_std = config.log_std
        # Initialize checkpoint arrays
        self.adp_iter_cp = np.array([0])
        self.adp_log_std_cp = np.array([self.log_std])
        self.adp_log_std_cp = np.pad(self.adp_log_std_cp, (0, self.adp_iter_cp.size - self.adp_log_std_cp.size), 'edge')
        self.adp_policy_lr_cp = np.array([self.policy_lr])
        self.adp_policy_lr_cp = np.pad(self.adp_policy_lr_cp, (0, self.adp_iter_cp.size - self.adp_policy_lr_cp.size), 'edge')
        # Initialize adaptive values
        self.adp_log_std = None
        self.adp_policy_lr = None

    def update_adaptive_params(self, i_iter):
        cp = self.adp_iter_cp
        ind = np.where(i_iter >= cp)[0][-1]
        nind = ind + int(ind < len(cp) - 1)
        t = (i_iter - self.adp_iter_cp[ind]) / (cp[nind] - cp[ind]) if nind > ind else 0.0
        self.adp_policy_lr = self.adp_policy_lr_cp[ind] * (1-t) + self.adp_policy_lr_cp[nind] * t
        if hasattr(self.config, 'fix_std') and self.config.fix_std:
            self.adp_log_std = self.adp_log_std_cp[ind] * (1-t) + self.adp_log_std_cp[nind] * t


# Add Argument Parser for introduction
def parse_args():
    parser = argparse.ArgumentParser()
    curr_dir = os.getcwd()
    yamlFile = curr_dir + "/assets/yaml/humanoidJump.yaml"
    
    parser.add_argument("--config", type=str, default=yamlFile,
                        help="Path to config YAML file")

    parser.add_argument("--train", type=bool, default=False,
                        help="Run training (default: True)")

    parser.add_argument("--no-train", dest='train', action='store_false',
                        help="Disable training")

    parser.add_argument("--eval", type=bool, default=False,
                        help="Run evaluation")
    
    parser.add_argument("--todo", type=str, default="all",
                        choices=["all", "mass", "torque", "ablation"],
                        help="Train specific TODO: all, mass, torque, ablation")
    parser.add_argument("--ablation-mode", type=str, default="full",
                        choices=["full", "no_imitation", "no_height", "no_stability", "no_distance", 
                                 "no_squat", "no_leg", "no_sym", "no_takeoff", "no_yaw", "no_upright"],
                        help="Specific ablation mode to train (if todo=ablation)")
    
    args = parser.parse_args()
    return args

def loadConfig(config, **kwargs):
    assert config is not None, "Enter a valid config file"
    
    config = OmegaConf.load(config)
    
    config.todo_settings = OmegaConf.create({
        "enable_mass_changes": True,
        "enable_torque_changes": True,
        "ablation_mode": None  # Will be overridden by args.ablation_mode if set
    })

    if kwargs:
        config.update(kwargs)
    
    return config

def train_loop(config, todo_mode="all", specific_ablation_mode=None):
    modelPath = config.model
    file = config.mocap_file
    curr_path = os.getcwd()
    filePath = curr_path + file
    modelPath = curr_path + modelPath

    # Set random seeds
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    env_seed = config.seed

    for index, mode in enumerate(trainingModes):
        # Filter modes based on todo_mode
        if todo_mode != "all":
            if todo_mode == "mass" and mode not in ["mass_changes", "mass_torque"]:
                continue
            elif todo_mode == "torque" and mode not in ["torque_changes", "mass_torque"]:
                continue
            elif todo_mode == "ablation" and specific_ablation_mode and mode != f"ablation_{specific_ablation_mode}":
                continue

        print(f"Training mode: {mode}")
        # Initialize environment
        env = HumanoidEnv(modelPath, filePath)
        eval_env = HumanoidEnv(modelPath, filePath)

        # Configure environment based on mode
        if mode.startswith("ablation_"):
            env.current_ablation_mode = mode.replace("ablation_", "")
            eval_env.current_ablation_mode = mode.replace("ablation_", "")
            env.enable_mass_changes = False
            env.enable_torque_changes = False
        elif mode == "mass_changes":
            env.enable_mass_changes = True
            env.enable_torque_changes = False
        elif mode == "torque_changes":
            env.enable_mass_changes = False
            env.enable_torque_changes = True
        elif mode == "mass_torque":
            env.enable_mass_changes = True
            env.enable_torque_changes = True
        else:  # jumpDistStable or jump
            env.enable_mass_changes = False
            env.enable_torque_changes = False

        # Set task mode
        env.set_mode = mode
        eval_env.set_mode = mode

        # Initialize agent
        agent = PPO_Agent(config)
        configUpdate = UpdateConfig(config)

        # Initialize TensorBoard writer
        from torch.utils.tensorboard import SummaryWriter
        date = datetime.date.today()
        time = datetime.datetime.now()
        timenow = f'_{date.day}_{date.month}__{time.hour}_{time.minute}_{index}'
        writepath = f'runsTesting/{mode}{timenow}'
        writer = SummaryWriter(log_dir=writepath)

        # Training loop
        acc_score = []
        traj_lenth = 0
        total_steps = 0
        i_iter = 0
        best = -np.inf    

        try:
            while total_steps < config.max_train_steps:
                i_iter += 1 
                configUpdate.update_adaptive_params(i_iter)
                set_optimizer_lr(agent.actor_optimizer, configUpdate.adp_policy_lr)

                env.set_mode = trainingModes[index]
                eval_env.set_mode = trainingModes[index]
                s, info = env.reset(seed=env_seed)
                env_seed += 1
                done = False
                while not done:
                    a, logprob_a = agent.select_action(s, deterministic=False)
                    act = Action_adapter(a, config.max_action)
                    
                    s_next, r, done, tr, _ = env.step(act)
                    dw = done or tr
                    agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx=traj_lenth)
                    s = s_next

                    traj_lenth += 1
                    total_steps += 1
                    acc_score.append(r)
                    
                    if traj_lenth % config.horizon == 0:
                        losses = agent.train()
                        traj_lenth = 0
                        logger(writer, losses, total_steps, env)
                            
                    # Print training progress
                    if total_steps % config.print_interval == 0:
                        avg_score = np.mean(acc_score)
                        rewardLogger(writer, avg_score, total_steps, env)
                        jointLogger(writer, total_steps, act, env) 
                        writer.add_scalar('train/max_distance', env.max_distance, global_step=total_steps)
                        writer.add_scalar('train/ablation_mode', env.ablation_modes.index(env.current_ablation_mode) if env.current_ablation_mode in env.ablation_modes else 0, global_step=total_steps)
                        acc_score = []
                    
                    if total_steps % config.eval_interval == 0:
                        score = evaluate_policy(eval_env, agent, config.max_action, turns=10)
                        writer.add_scalar('eval/episode_reward', score, global_step=total_steps)
                        writer.add_scalar('eval/max_distance', eval_env.max_distance, global_step=total_steps)
                        print(f'Mode {mode}, Evaluation at step {total_steps // 1000}k: avg reward = {score:.3f}, max distance = {eval_env.max_distance:.3f}')
                        if score > best:
                            best = score
                            agent.save(f'{mode}_best', 0, folder=f"modelTest/{mode}")
                            np.save(f"./modelTest/{mode}/norm/{mode}_normalizer_best{0}.npy", agent.obs_normalizer.rms.save_variables())
                    
                    if total_steps % config.save_interval == 0:
                        agent.save(mode, total_steps // 1000, folder=f"modelTest/{mode}")
                        np.save(f"./modelTest/{mode}/norm/{mode}_normalizer_{total_steps // 1000}.npy", agent.obs_normalizer.rms.save_variables())

                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print(f'Training interrupted for mode {mode}. Saving model...')
            agent.save(mode, total_steps // 1000, folder=f"modelTest/{mode}")
            np.save(f"./modelTest/{mode}/norm/{mode}_normalizer_{total_steps // 1000}.npy", agent.obs_normalizer.rms.save_variables())

        finally:
            env.close()
            eval_env.close()
            writer.close()

from PIL import Image, ImageDraw

def make_transparent(img, alpha=180):
    img = img.convert("RGBA")
    r, g, b, _ = img.split()
    alpha_layer = Image.new("L", img.size, alpha)
    return Image.merge("RGBA", (r, g, b, alpha_layer))

def create_motion_trail(env, agent, max_action, mode, turns=1, save_every=20, step_x=2000):
    render_w, render_h = 2048, 2048
    env.fixed_start = True

    for j in range(turns):
        s, _ = env.reset()
        done = False
        frames = []
        t = 0
        max_steps = 121

        while not done and t < max_steps:
            a, _ = agent.select_action(s, deterministic=True)
            act = Action_adapter(a, max_action)
            s_next, r, done, tr, info = env.step(act)
            s = s_next

            if t % save_every == 0:
                img = env.render()
                pil_img = Image.fromarray(img).convert("RGBA")
                frames.append(pil_img)
                print(f"Saved frame at step {t}")
            t += 1

        num_frames = len(frames)
        offset_x = step_x  # pixels to shift each frame rightward
        total_width = render_w + offset_x * (num_frames - 1)
        canvas = Image.new("RGBA", (total_width, render_h), (0, 0, 0, 0))

        for i, frame in enumerate(frames):
            frame = frame.resize((render_w, render_h))
            translucent = make_transparent(frame, alpha=255)
            x = i * offset_x
            canvas.alpha_composite(translucent, (x, 0))

        canvas.save(f"{mode}_motion_progression.png")
        print(f"Saved {mode}_motion_progression.png with {num_frames} frames")

    return r

# Load environment and config yaml file
def main():
    args = parse_args()
    config = loadConfig(args.config)
    
    if args.eval or not args.train:
        mode = trainingModes[3] if args.todo == "all" else args.todo
        if args.todo == "ablation" and args.ablation_mode:
            mode = f"ablation_{args.ablation_mode}"
        config.render = args.eval
        # Run a single mode for evaluation
        modelPath = config.model
        file = config.mocap_file
        curr_path = os.getcwd()
        filePath = curr_path + file
        modelPath = curr_path + modelPath
        print(f"Eval Mode: {mode}")
        env = HumanoidEnv(modelPath, filePath, render_mode="human" if args.eval else None)
        agent = PPO_Agent(config)
        # for k in range(500, 30001, 500):
            # config.model_index = k
            # print(f"Iteration number: {k}")
        agent.load(mode, config.model_index, folder=f"modelTest/{mode}")
        normalizer_state = np.load(f"./modelTest/{mode}/norm/{mode}_normalizer_{config.model_index}.npy", allow_pickle=True).item()
        agent.obs_normalizer.rms.load_variables(normalizer_state)
        if config.best == True:
            agent.load(f'{mode}_best', 0, folder=f"modelTest/{mode}")
            normalizer_state = np.load(f"./modelTest/{mode}/norm/{mode}_normalizer_best{0}.npy", allow_pickle=True).item()
            agent.obs_normalizer.rms.load_variables(normalizer_state)
        if config.capture:
            env = HumanoidEnv(modelPath, filePath, render_mode="rgb_array")
            env.set_mode = mode
            if mode.startswith("ablation_"):
                env.current_ablation_mode = mode.replace("ablation_", "")
            ep_r = create_motion_trail(env, agent, mode=mode, max_action=config.max_action, turns=1)
            print(f'Episode Iteration Passed')
        else:
            env.set_mode = mode
            if mode.startswith("ablation_"):
                env.current_ablation_mode = mode.replace("ablation_", "")
            ep_r = evaluate_policy_metrics(env, agent, max_action=config.max_action, turns=150)
            # ep_r = evaluate_policy(env, agent, max_action=config.max_action, turns=1)
            print(f'Mode {mode}, Episode Reward: {ep_r}')
    else:
        # Train all modes sequentially
        train_loop(config, todo_mode=args.todo, specific_ablation_mode=args.ablation_mode)
        
if __name__ == "__main__":
    main()
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
from environment.humanoidEnv import HumanoidEnv
from algorithm.ppo import PPO_Agent
from utils.ppoUtils import evaluate_policy, Action_adapter, logger, rewardLogger, jointLogger
# from utils.sacUtils import Action_adapter

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

    parser.add_argument("--train", action='store_true', default=True,
                        help="Run training (default: True)")

    parser.add_argument("--no-train", dest='train', action='store_false',
                        help="Disable training")

    parser.add_argument("--eval", action='store_true', default=False,
                        help="Run evaluation")
    args = parser.parse_args()
    return args

def loadConfig(config, **kwargs):
    assert config is not None, "Enter a valid config file"
    
    config = OmegaConf.load(config)
    
    if kwargs:
        config.update(kwargs)
    
    return config

trainingModes = ["jumpReset", "jumpMimicReset", "jumpResetInit_ModMimic", "jumpResetInit_Mod"]

def train_loop(config):
    modelPath = config.model
    file = config.mocap_file
    curr_path = os.getcwd()
    filePath = curr_path + file
    modelPath = curr_path + modelPath

    env = HumanoidEnv(modelPath, filePath)
    eval_env = HumanoidEnv(modelPath, filePath)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    env_seed = config.seed
    
    agent = PPO_Agent(config)
    # Initialize adaptive parameter tuning
    configUpdate = UpdateConfig(config)
    if config.render:
        index = 2
        if config.best == True:
            agent.load(f"{trainingModes[index]}_best", 0)
            normalizer_state = np.load(f"./model/norm/{trainingModes[index]}_normalizer_best{0}.npy", allow_pickle=True).item()
            agent.obs_normalizer.rms.load_variables(normalizer_state)
        else:
            agent.load(f"{trainingModes[index]}", config.model_index)
            normalizer_state = np.load(f"./model/norm/{trainingModes[index]}_normalizer_{config.model_index}.npy", allow_pickle=True).item()
            agent.obs_normalizer.rms.load_variables(normalizer_state)

        env = HumanoidEnv(modelPath, filePath, render_mode="human")
        while True:
            env.set_mode = trainingModes[index]
            ep_r = evaluate_policy(env, agent, max_action=config.max_action, turns=1)
            print(f'Episode Reward: {ep_r}')
    else:
        for index in range(len(trainingModes)):
            print(f"Current training mode: {trainingModes[index]}")
            from torch.utils.tensorboard import SummaryWriter
            date = datetime.date.today()
            time = datetime.datetime.now()
            timenow = f'_{date.day}_{date.month}__{time.hour}_{time.minute}'
            writepath = f'runs/{trainingModes[index]}{timenow}'
            writer = SummaryWriter(log_dir=writepath)
            
            acc_score = []
            traj_lenth = 0
            total_steps = 0
            i_iter = 0  # Track iterations for adaptive tuning
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
                            acc_score = []
                        
                        # Periodic evaluation
                        if total_steps % config.eval_interval == 0:
                            score = evaluate_policy(eval_env, agent, config.max_action, turns=10)
                            writer.add_scalar('eval/episode_reward', score, global_step=total_steps)
                            print(f'Evaluation at step {total_steps // 1000}k: avg reward = {score:.3f}')
                            if score > best:
                                best = score
                                agent.save(f"{trainingModes[index]}_best", 0)
                                np.save(f"./model/norm/{trainingModes[index]}_normalizer_best{0}.npy", agent.obs_normalizer.rms.save_variables())
                        
                        # Periodic save
                        if total_steps % config.save_interval == 0:
                            agent.save(f"{trainingModes[index]}", total_steps // 1000)
                            np.save(f"./model/norm/{trainingModes[index]}_normalizer_{total_steps // 1000}.npy", agent.obs_normalizer.rms.save_variables())

                    torch.cuda.empty_cache()

            except KeyboardInterrupt:
                print('Training interrupted. Saving model...')
                agent.save(f"{trainingModes[index]}", total_steps // 1000)
                np.save(f"./model/norm/{trainingModes[index]}_normalizer_{total_steps // 1000}.npy", agent.obs_normalizer.rms.save_variables())

            finally:
                env.close()
                eval_env.close()

# Load environment and config yaml file
def main():
    args = parse_args()
    config = loadConfig(args.config)
    train_loop(config)
        
if __name__ == "__main__":
    main()
from utils.ppoUtils import GaussianActor, Critic, Normalizer
import numpy as np
import copy
import torch
import math


class PPO_Agent(object):
    def __init__(self, config):
        self.config = config
        # Choose distribution for the actor
        self.actor = GaussianActor(self.config.state_dim, self.config.action_dim, self.config.net_width, self.config.hidden_dim).to(self.config.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.a_lr)

        # Build Critic
        self.critic = Critic(self.config.state_dim, self.config.net_width, self.config.hidden_dim).to(self.config.dvc)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.c_lr)

        # Build Trajectory holder
        self.s_hoder = np.zeros((self.config.horizon, self.config.state_dim),dtype=np.float32)
        self.a_hoder = np.zeros((self.config.horizon, self.config.action_dim),dtype=np.float32)
        self.r_hoder = np.zeros((self.config.horizon, 1),dtype=np.float32)
        self.s_next_hoder = np.zeros((self.config.horizon, self.config.state_dim),dtype=np.float32)
        self.logprob_a_hoder = np.zeros((self.config.horizon, self.config.action_dim),dtype=np.float32)
        self.done_hoder = np.zeros((self.config.horizon, 1),dtype=np.bool_)
        self.dw_hoder = np.zeros((self.config.horizon, 1),dtype=np.bool_)

        self.entropy_coef = self.config.entropy_coef
        self.obs_normalizer = Normalizer(self.config.state_dim)

    def select_action(self, state, deterministic):
        with torch.no_grad():
            state = self.obs_normalizer(state)
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.config.dvc)
            if deterministic:
                # only used when evaluate the policy. Making the performance more stable
                a = self.actor.deterministic_act(state)
                return a.cpu().numpy()[0], None  # action is in shape (adim, 0)
            else:
                # only used when interact with the env
                dist = self.actor.get_dist(state)
                a = dist.sample()
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
                a = torch.clamp(a, 0, 1)
                return a.cpu().numpy()[0], logprob_a # both are in shape (adim, 0)


    def train(self):
        self.entropy_coef*=self.config.entropy_coef_decay

        '''Prepare PyTorch data from Numpy data'''
        s = torch.from_numpy(self.s_hoder).to(self.config.dvc)
        if torch.isnan(s).any() or torch.isinf(s).any():
            print("NaN or inf in input states:", s)
            raise ValueError("Invalid input states")
        a = torch.from_numpy(self.a_hoder).to(self.config.dvc)
        r = torch.from_numpy(self.r_hoder).to(self.config.dvc)
        s_next = torch.from_numpy(self.s_next_hoder).to(self.config.dvc)
        logprob_a = torch.from_numpy(self.logprob_a_hoder).to(self.config.dvc)
        done = torch.from_numpy(self.done_hoder).to(self.config.dvc)
        dw = torch.from_numpy(self.dw_hoder).to(self.config.dvc)

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            '''dw for TD_target and Adv'''
            deltas = r + self.config.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.config.gamma * self.config.lambd * adv[-1] * (~mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.config.dvc)
            td_target = adv + vs
            adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps
            
        actor_loss, critic_loss = [], []
        """Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
        a_optim_iter_num = int(math.ceil(s.shape[0] / self.config.a_optim_batch_size))
        c_optim_iter_num = int(math.ceil(s.shape[0] / self.config.c_optim_batch_size))
        for i in range(self.config.n_epochs):

            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.config.dvc)
            s, a, td_target, adv, logprob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()
            
            '''update the actor'''
            for i in range(a_optim_iter_num):
                index = slice(i * self.config.a_optim_batch_size, min((i + 1) * self.config.a_optim_batch_size, s.shape[0]))
                distribution = self.actor.get_dist(s[index])
                dist_entropy = distribution.entropy().sum(1, keepdim=True)
                logprob_a_now = distribution.log_prob(a[index])
                ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_rate, 1 + self.config.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()
                actor_loss.append(a_loss.mean().item())

            '''update the critic'''
            for i in range(c_optim_iter_num):
                index = slice(i * self.config.c_optim_batch_size, min((i + 1) * self.config.c_optim_batch_size, s.shape[0]))
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name,param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.config.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 40)
                self.critic_optimizer.step()
                critic_loss.append(c_loss.item())
        
        return {
            'actor_loss': np.mean(actor_loss),
            'critic_loss': np.mean(critic_loss),
            'adv_mean': adv.mean().item(),
            'adv_std': adv.std().item(),
        }

    def put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
        s = self.obs_normalizer(s)
        s_next = self.obs_normalizer(s_next)
        if np.isnan(s).any() or np.isinf(s).any() or np.isnan(s_next).any() or np.isinf(s_next).any():
            print("NaN or inf in state or next state:", s, s_next)
            raise ValueError("Invalid state detected")
        if np.isnan(r) or np.isinf(r):
            print("NaN or inf in reward:", r)
            raise ValueError("Invalid reward detected")
        self.s_hoder[idx] = s
        self.a_hoder[idx] = a
        self.r_hoder[idx] = r
        self.s_next_hoder[idx] = s_next
        self.logprob_a_hoder[idx] = logprob_a
        self.done_hoder[idx] = done
        self.dw_hoder[idx] = dw

    def save(self,EnvName, timestep, folder="model"):
        torch.save(self.actor.state_dict(), "./{}/actor/{}_actor{}.pth".format(folder, EnvName,timestep))
        torch.save(self.critic.state_dict(), "./{}/critic/{}_q_critic{}.pth".format(folder, EnvName,timestep))

    def load(self,EnvName, timestep, folder="model"):
        self.actor.load_state_dict(torch.load("./{}/actor/{}_actor{}.pth".format(folder, EnvName, timestep), map_location=self.config.dvc))
        self.critic.load_state_dict(torch.load("./{}/critic/{}_q_critic{}.pth".format(folder, EnvName, timestep), map_location=self.config.dvc))



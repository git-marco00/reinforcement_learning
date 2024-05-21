import gymnasium as gym
from my_utils.Replay_buffer import ReplayBuffer
from networks.stochastic_network import Actor, Critic
from tqdm import trange
import torch
from matplotlib import pyplot as plt
import os
import numpy as np
from torch.distributions.categorical import Categorical


class PGO_trainer():
    def __init__(self, env, gamma, episodes, actor_lr, critic_lr, buffer_max_len, steps2opt, steps2converge, mode, tau, batch_size=64, critic_bootstrap=10):
        self.env = env
        self.gamma = gamma
        self.episodes = episodes
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.buffer_max_len = buffer_max_len
        self.steps2opt = steps2opt
        self.batch_size = batch_size
        self.steps2converge = steps2converge
        self.mode = mode
        self.tau = tau
        self.critic_bootstrap = critic_bootstrap

        self.critic_loss_history = []
        self.actor_loss_history = []
        self.mean_history = []
        self.std_history = []
        self.scores = []

        self.n_actions = 2
        self.n_states = 4
        
        self.replay_buffer = ReplayBuffer(maxlen=buffer_max_len)

        # ACTOR
        self.actor_network = Actor(self.n_states, self.n_actions)
        self.actor_optim = torch.optim.Adam(params = self.actor_network.parameters(), lr = actor_lr)
        
        # CRITIC with V
        self.critic_network = Critic(input_dim=self.n_states, output_dim=1)
        self.critic_optim = torch.optim.Adam(params = self.critic_network.parameters(), lr = self.critic_lr)

        if self.mode == "target_networks":
            self.target_critic_network = Critic(input_dim=self.n_states, output_dim=1)
            self.target_critic_network.load_state_dict(self.critic_network.state_dict())

            self.target_actor_network = Actor(n_states=self.n_states, n_actions=self.n_actions)
            self.target_actor_network.load_state_dict(self.actor_network.state_dict())
       
        
        self.loss_fn = torch.nn.MSELoss()
        self.ep = 0

        
    def train(self):
        t = trange(self.episodes)
        for self.ep in t:
            step = 0
            truncated = False
            terminated = False

            state = self.env.reset()[0]
            state = torch.tensor(state)

            self.scores.append(0)

            while not (truncated or terminated):
                if self.mode == "target_networks": distr_params = self.target_actor_network(state.unsqueeze(0))
                else: distr_params = self.actor_network(state.unsqueeze(0))

                distr = Categorical(distr_params)
			
                action = distr.sample().squeeze()	# returns a tenso

                new_state, reward, terminated, truncated, _ = self.env.step(action.numpy())	# tensor => numpy array

                self.replay_buffer.append((state, action, new_state, reward, terminated, truncated))

                self.scores[-1]+=reward

                step += 1
                
                state = torch.tensor(new_state)
    
                if step % self.steps2opt == 0 and len(self.replay_buffer) > self.batch_size*2:
                    self.optimize()
                        
                if self.mode == "target_networks" and step % self.steps2converge == 0:
                    self.update_target_model(tau = self.tau, target_network= self.target_critic_network, actual_network=self.critic_network)
                    self.update_target_model(tau = self.tau, target_network= self.target_actor_network, actual_network=self.actor_network)


            t.set_description(f"Episode score: {round(self.scores[-1], 2)}, critic_loss: {round(sum(self.critic_loss_history)/(len(self.critic_loss_history)+1),2)}, actor_loss: {round(sum(self.actor_loss_history)/(len(self.actor_loss_history)+1),2)}")
        
  
    def optimize(self):
        batch = self.replay_buffer.sample(sample_size=self.batch_size)
        
        with torch.no_grad():
            states = [x[0] for x in batch]
            actions = [x[1] for x in batch]
            next_states = [torch.tensor(x[2]) for x in batch]
            rewards = [torch.tensor(x[3], dtype=torch.float32) for x in batch]
            terminated = [torch.tensor(x[4]) for x in batch]
            
            actions_batch = torch.stack(actions)
            states_batch = torch.stack(states)
            next_states_batch = torch.stack(next_states)
            rewards_batch = torch.stack(rewards)
            terminated_batch = torch.stack(terminated)      

        ############# CRITIC OPTIMIZATION #############

        # current batch
        pred_v_batch = self.critic_network(states_batch).squeeze()

        # target
        with torch.no_grad():   
            if self.mode == "target_networks": next_v = self.target_critic_network(next_states_batch).squeeze()
            else: next_v = self.critic_network(next_states_batch).squeeze()     
            target_v_batch = rewards_batch + ~terminated_batch * self.gamma * next_v

        critic_loss = self.loss_fn(pred_v_batch, target_v_batch)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.critic_loss_history.append(critic_loss.item())

        ############# ACTOR OPTIMIZATION #############
        if self.ep > self.critic_bootstrap:
            distr_params = self.actor_network(states_batch)
            m = Categorical(distr_params)
            log_probs = m.log_prob(actions_batch)

            pred_v_batch = self.critic_network(states_batch).squeeze()
            with torch.no_grad():   
                if self.mode == "target_networks": next_v = self.target_critic_network(next_states_batch).squeeze()
                else: next_v = self.critic_network(next_states_batch).squeeze()              
                target_v_batch = rewards_batch + ~terminated_batch * self.gamma * next_v
            delta_t = (target_v_batch - pred_v_batch).detach()

            actor_loss = - (delta_t * log_probs).mean()

            self.actor_optim.zero_grad()
            self.actor_loss_history.append(actor_loss.mean().item())
            actor_loss.backward()
            self.actor_optim.step()


    def plot(self, plot_scores=True, plot_critic_loss=True, plot_actor_loss=True):
        PATH = os.path.abspath(__file__)

        if plot_scores is True:
            window_size = 20
            smoothed_data = np.convolve(self.scores, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            plt.savefig("PGO/res/A2C/tn_scores.png")
            plt.clf()

        if plot_critic_loss is True:
            window_size = 1000
            smoothed_data = np.convolve(self.critic_loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            plt.savefig("PGO/res/A2C/tn_critic_loss.png")
            plt.clf()

        if plot_actor_loss is True:
            window_size = 1000
            smoothed_data = np.convolve(self.actor_loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            plt.savefig("PGO/res/A2C/tn_actor_loss.png")
            plt.clf()

        
    def update_target_model(self, tau, target_network, actual_network):
        weights = actual_network.state_dict()
        target_weights = target_network.state_dict()
        for i in target_weights:
            target_weights[i] = weights[i] * tau + target_weights[i] * (1-tau)
        target_network.load_state_dict(target_weights)
   

# PARAMETERS
env = gym.make('CartPole-v1')
gamma = 0.99
episodes = 500
actor_lr = 0.001
critic_lr= 0.001
buffer_max_len = 500000
steps2opt = 5
steps2converge = 10
mode = "target_networks"
batch_size = 32
tau = 0.2
critic_bootstrap = 30

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


trainer = PGO_trainer(env = env, gamma=gamma, episodes=episodes, actor_lr=actor_lr, critic_lr=critic_lr, buffer_max_len=buffer_max_len, steps2opt=steps2opt, steps2converge=steps2converge, mode=mode, batch_size=batch_size, tau = tau, critic_bootstrap=critic_bootstrap)
trainer.train()
trainer.plot(plot_scores=True, plot_actor_loss=True, plot_critic_loss=True)
env.close()
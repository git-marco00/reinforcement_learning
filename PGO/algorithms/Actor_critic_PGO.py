import gymnasium as gym
from my_utils.Replay_buffer import ReplayBuffer
from networks.actor_network import Actor
from networks.critic_network import Critic
import random
from tqdm import trange
import torch
from matplotlib import pyplot as plt
import os
import numpy as np


class PGO_trainer():
    def __init__(self, env, gamma, episodes, actor_lr, critic_lr, buffer_max_len, steps2opt, steps2converge, mode, target_model_update_rate = 1e-2, batch_size=64):
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
        self.target_model_update_rate = target_model_update_rate

        self.critic_loss_history = []
        self.actor_loss_history = []
        self.scores = []

        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.shape[0]
        self.replay_buffer = ReplayBuffer(maxlen=buffer_max_len)

        # ACTOR
        self.actor_network = Actor(self.n_states, self.n_actions)
        self.actor_optim = torch.optim.AdamW(params = self.actor_network.parameters(), lr = actor_lr, amsgrad = True)

        # CRITIC
        self.critic_network = Critic(n_actions=self.n_actions, n_states=self.n_states)
        self.target_critic_network = Critic(n_actions=self.n_actions, n_states=self.n_states)
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())
        self.critic_optim = torch.optim.AdamW(params = self.critic_network.parameters(), lr = self.critic_lr, amsgrad = True)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
    def train(self):
        eps = 0
        for ep in trange(self.episodes):
            truncated = False
            terminated = False

            state = self.env.reset()[0]
            state = torch.tensor(state)
            state[5]*=2.5
            episode_score = 0
            step = 0

            if ep > 300:
                self.critic_optim = torch.optim.AdamW(params = self.critic_network.parameters(), lr = 1e-5, amsgrad = True)

            while not (truncated or terminated):
                action = self.actor_action_selection(state.unsqueeze(0), eps).squeeze()

                #action = self.critic_action_selection(state.unsqueeze(0), eps)   
                
                new_state, reward, terminated, truncated, _ = self.env.step(action.item())

                self.replay_buffer.append((state, action, new_state, reward, terminated, truncated))

                episode_score += reward
            
                step += 1
                    
                state = torch.tensor(new_state)
                state[5]*=2.5

                if step % self.steps2opt == 0 and len(self.replay_buffer) > self.batch_size*2:
                    for i in range(2):
                        self.optimize_critic()
                        self.optimize_actor()
                        self.update_target_model(self.target_model_update_rate)

                if step % self.steps2converge == 0:
                    pass
            
            self.scores.append(episode_score)
            
            if eps > 0.01:
                eps *= 0.992
            
    def actor_action_selection(self, state, eps):
        if random.random() <= eps:
            action = torch.tensor(self.env.action_space.sample())
        else:
            with torch.no_grad():
                probs = self.actor_network(state)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
        return action
        
    def critic_action_selection(self, state, eps):
        if random.random() <= eps:
            action = torch.tensor(self.env.action_space.sample())
        else:
            with torch.no_grad():
                action = torch.argmax(self.critic_network(state))
        return action
        
    def optimize_actor(self):
        batch = self.replay_buffer.sample(sample_size=self.batch_size)

        states = [x[0] for x in batch]
        actions = [x[1] for x in batch]

        states_batch = torch.stack(states)
        actions_batch = torch.stack(actions)

        probs = self.actor_network(states_batch)
        dist = torch.distributions.Categorical(probs)
        log_probs_batch = dist.log_prob(actions_batch)

        with torch.no_grad():
            q_values_all_actions = self.critic_network(states_batch)
            q_values = torch.gather(q_values_all_actions, 1, actions_batch.unsqueeze(dim=1)).squeeze()
    
        loss = - log_probs_batch * q_values

        self.actor_optim.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=1.0)
        self.actor_loss_history.append(loss.mean().item())
        self.actor_optim.step()   
  
    def optimize_critic(self):
        batch = self.replay_buffer.sample(sample_size=self.batch_size)

        states = [x[0] for x in batch]
        actions = [x[1] for x in batch]
        next_states = [torch.tensor(x[2]) for x in batch]
        rewards = [torch.tensor(x[3], dtype=torch.float32) for x in batch]
        terminated = [torch.tensor(x[4]) for x in batch]
        
        actions_batch = torch.stack(actions).unsqueeze(dim=1)
        states_batch = torch.stack(states)
        next_states_batch = torch.stack(next_states)
        rewards_batch = torch.stack(rewards)
        terminated_batch = torch.stack(terminated)

        # Q(s,a,w) = reward + gamma * max a' Q(s', a', w-) - Q(s, a, w)
        
        # current batch
        q_batch = self.critic_network(states_batch)
        current_batch = torch.gather(q_batch, 1, actions_batch).squeeze() # Q(s, a, w)

        # target
        if self.mode == "simple_DQN":
            with torch.no_grad():
                target_q = self.target_critic_network(next_states_batch)
                max_q,_ = torch.max(target_q, dim=1)
                target_batch = rewards_batch + ~terminated_batch * self.gamma * max_q    # reward + gamma * max a' Q(s', a', w-)
        
        elif self.mode == "double_DQN":
            with torch.no_grad():
                actual_q = self.critic_network(next_states_batch)
                max_actual_actions_batch = torch.argmax(actual_q, dim=1).unsqueeze(1)
                target_q = self.target_critic_network(next_states_batch)
                best_q = torch.gather(target_q, 1, max_actual_actions_batch).squeeze()
                target_batch = rewards_batch + ~terminated_batch * self.gamma * best_q


        loss = self.loss_fn(current_batch, target_batch)
        self.critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=1.0)
        self.critic_optim.step()
        self.critic_loss_history.append(loss.item())

    def plot(self, plot_scores=True, plot_critic_loss=True, plot_actor_loss=True):
        PATH = os.path.abspath(__file__)

        if plot_scores is True:
            window_size = 20
            smoothed_data = np.convolve(self.scores, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            plt.savefig("PGO/res/PGO_AC_DDQN_score_mine.png")
            plt.clf()

        if plot_critic_loss is True:
            window_size = 1000
            smoothed_data = np.convolve(self.critic_loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            plt.savefig("PGO/res/PGO_AC_DDQN_critic_loss_mine.png")
            plt.clf()

        if plot_actor_loss is True:
            window_size = 1000
            smoothed_data = np.convolve(self.actor_loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            plt.savefig("PGO/res/PGO_AC_DDQN_actor_loss_mine.png")
            plt.clf()
        
    def update_target_model(self, tau):
        weights = self.critic_network.state_dict()
        target_weights = self.target_critic_network.state_dict()
        for i in target_weights:
            target_weights[i] = weights[i] * tau + target_weights[i] * (1-tau)
        self.target_critic_network.load_state_dict(target_weights)
   

# PARAMETERS
env = gym.make('LunarLander-v2')
gamma = 0.99
episodes = 1000
actor_lr = 1e-5
critic_lr=1e-3
buffer_max_len = 100000
steps2opt = 2
steps2converge = 4
mode = "simple_DQN"
batch_size = 64
target_model_update_rate = 5e-3


trainer = PGO_trainer(env = env, gamma=gamma, episodes=episodes, actor_lr=actor_lr, critic_lr=critic_lr, buffer_max_len=buffer_max_len, steps2opt=steps2opt, steps2converge=steps2converge, mode=mode, batch_size=batch_size, target_model_update_rate = target_model_update_rate)
trainer.train()
trainer.plot(plot_scores=True, plot_actor_loss=True, plot_critic_loss=True)
env.close()



    

    

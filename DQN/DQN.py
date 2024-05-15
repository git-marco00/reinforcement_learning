import gymnasium as gym
from my_utils.Replay_buffer import ReplayBuffer
from my_utils.Q_Network import Q_network
import random
from tqdm import trange
import torch
from matplotlib import pyplot as plt
import os
import numpy as np


class PGO_trainer():
    def __init__(self, env, gamma, episodes, lr, buffer_max_len, steps2opt, steps2converge, mode, eps_decay, eps_end, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.episodes = episodes
        self.lr = lr
        self.buffer_max_len = buffer_max_len
        self.steps2opt = steps2opt
        self.batch_size = batch_size
        self.steps2converge = steps2converge
        self.mode = mode
        self.eps = 1
        self.eps_decay = eps_decay
        self.eps_end = eps_end

        self.loss_history = []
        self.scores = []
        
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.shape[0]
        self.replay_buffer = ReplayBuffer(maxlen=buffer_max_len)

        self.Q_network = Q_network(n_states=self.n_states, n_actions=self.n_actions)
        self.target_network = Q_network(n_states=self.n_states, n_actions=self.n_actions)
        self.target_network.load_state_dict(self.Q_network.state_dict())
        self.optim = torch.optim.Adam(self.Q_network.parameters(), lr = self.lr)
        self.loss = torch.nn.MSELoss()
        
    def train(self):
        step = 0
        t = trange(self.episodes)
        for ep in t:
            truncated = False
            terminated = False

            state = self.env.reset()[0]
            state = torch.tensor(state)

            episode_score = 0
            while not (truncated or terminated):
                if random.random() <= self.eps:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        actions = self.Q_network(state.unsqueeze(0)).squeeze()
                        action = torch.argmax(actions).item()

                new_state, reward, terminated, truncated, _ = self.env.step(action) 

                self.replay_buffer.append((state, action, new_state, reward, terminated, truncated))

                episode_score += reward
            
                step += 1
                    
                state = torch.tensor(new_state)

                if step % self.steps2opt == 0 and len(self.replay_buffer) > self.batch_size*2:
                    self.optimize()

                if step % self.steps2converge == 0:
                    self.target_network.load_state_dict(self.Q_network.state_dict())
            
            self.scores.append(episode_score)

            t.set_description(f"Episode score: {round(episode_score, 2)}")

            self.update_eps()
            
    def update_eps(self):
        if self.eps > self.eps_end:
            self.eps *= self.eps_decay

  
    def optimize(self):
        batch = self.replay_buffer.sample(sample_size=self.batch_size)

        states = [x[0] for x in batch]
        actions = [torch.tensor(x[1]) for x in batch]
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
        q_batch = self.Q_network(states_batch)
        current_batch = torch.gather(q_batch, 1, actions_batch).squeeze() # Q(s, a, w)

        # target
        if self.mode == "simple_DQN":
            with torch.no_grad():
                target_q = self.target_network(next_states_batch)
                max_q,_ = torch.max(target_q, dim=1)
                target_batch = rewards_batch + ~terminated_batch * self.gamma * max_q    # reward + gamma * max a' Q(s', a', w-)
        
        elif self.mode == "double_DQN":
            with torch.no_grad():
                actual_q = self.Q_network(next_states_batch)
                max_actual_actions_batch = torch.argmax(actual_q, dim=1).unsqueeze(1)
                target_q = self.target_network(next_states_batch)
                best_q = torch.gather(target_q, 1, max_actual_actions_batch).squeeze()
                target_batch = rewards_batch + ~terminated_batch * self.gamma * best_q

        loss = self.loss(current_batch, target_batch)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.loss_history.append(loss.item())

    def plot(self, plot_scores=True, plot_loss=True):
        PATH = os.path.abspath(__file__)

        if plot_scores is True:
            window_size = 20
            smoothed_data = np.convolve(self.scores, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            plt.savefig("DQN/res/DDQN_score.png")
            plt.clf()

        if plot_loss is True:
            window_size = 1000
            smoothed_data = np.convolve(self.loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            plt.savefig("DQN/res/DDQN_loss.png")
            plt.clf()


# PARAMETERS
env = gym.make('LunarLander-v2')
gamma = 0.99
episodes = 2000
lr = 5e-4
buffer_max_len = int(1e5)
steps2opt = 4
steps2converge = 4
mode = "double_DQN"
batch_size = 64
eps_decay = 0.995
eps_end = 0.01

trainer = PGO_trainer(env = env, gamma=gamma, episodes=episodes, lr = lr, buffer_max_len=buffer_max_len, steps2opt=steps2opt, steps2converge=steps2converge, mode=mode, batch_size=batch_size, eps_decay=eps_decay, eps_end=eps_end)
trainer.train()
trainer.plot()
env.close()

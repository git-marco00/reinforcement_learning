import gymnasium as gym
import random
from tqdm import trange
from matplotlib import pyplot as plt
import os
import numpy as np

class TD_trainer():
    def __init__(self, env, mode, episodes, gamma, eps_start=1, eps_decay=0.95, eps_min=0.01):
        self.env = env
        self.mode = mode
        self.episodes = episodes
        self.gamma = gamma
        self.eps = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self.n_actions = self.env.action_space.n
        self.n_states = self.env.observation_space.n
        self.Q_s_a = np.zeros((self.n_states, self.n_actions))

        self.scores = []

    def train(self):
        for ep in trange(self.episodes):
            truncated = False
            terminated = False
            state = self.env.reset()[0]
            self.scores.append(0)
            
            while not (truncated or terminated):
                # eps greedy
                if self.eps > random.random():
                    action = self.env.action_space.sample()
                else:
                    action = self.Q_s_a[state].argmax()

                new_state, reward, terminated, truncated, _ = self.env.step(action)

                step = (state, action, reward, new_state)

                self.scores[ep]+=reward

                self.optimize(step)

                state = new_state

            self.eps = 1-ep/self.episodes

    def optimize(self, step):
        state, action, reward, new_state = step
        is_random = None
        if self.eps > random.random():
            new_action = self.env.action_space.sample()
            is_random = True
        else:
            new_action = self.Q_s_a[new_state].argmax()
            is_random = False

        # update rules
        if self.mode == "SARSA":
            self.Q_s_a[state, action] += self.eps*(reward + self.gamma*self.Q_s_a[new_state, new_action] - self.Q_s_a[state, action])

        elif self.mode == "Expected_SARSA":
            if is_random:
                pi_ap_sp = 1/self.n_actions
                sum = 0
                for act in range(self.n_actions):
                    sum += pi_ap_sp*self.Q_s_a[new_state, act]
            
            if not is_random:
                sum = self.Q_s_a[new_state, new_action]

            self.Q_s_a[state, action] += self.eps*(reward + self.gamma*sum - self.Q_s_a[state, action])

        elif self.mode == "Q_learning":
            pi_next_action = self.Q_s_a[new_state].argmax()
            self.Q_s_a[state, action] += self.eps * (reward + self.gamma * self.Q_s_a[new_state, pi_next_action]- self.Q_s_a[state, action])

    def plot_rewards(self):
        PATH = os.path.abspath(__file__)
        x = range(0, self.episodes)
        plt.plot(x, self.scores)
        plt.savefig(f"tabular_methods/TD/res/{self.mode}_{self.env.unwrapped.spec.id}.png")
    
    def test(self, env, n_episodes=5):
        for i in range (n_episodes):
            state = env.reset()[0]
            truncated = False
            terminated = False
            while not (truncated or terminated):
                action = self.Q_s_a[state].argmax()
                state, reward, terminated, truncated, _ = env.step(action)

env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode=None)
trainer = TD_trainer(env = env, gamma=0.99, episodes=1000, mode="Q_learning")
trainer.train()
trainer.plot_rewards()
env.close()
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="human")
trainer.test(env, n_episodes=1)
env.close()

        

                



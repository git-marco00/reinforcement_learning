import gymnasium as gym
import random
from tqdm import trange
from matplotlib import pyplot as plt
import os
import numpy as np

class MC_trainer():
    def __init__(self, env, mode, episodes, gamma):
        self.env = env
        self.mode = mode
        self.episodes = episodes
        self.gamma = gamma

        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.N_s_a = np.zeros((self.n_states, self.n_actions))
        self.Q_s_a = np.zeros((self.n_states, self.n_actions))
        self.scores = []

    def train(self):
        eps = 1
        steps = []

        # collect experience
        for i in trange(self.episodes):
            state = self.env.reset()[0]
            truncated = False
            terminated = False
            self.scores.append(0)
            while not (truncated or terminated):
                if eps > random.random():
                    action = self.env.action_space.sample()
                else:
                    action = self.Q_s_a[state].argmax()

                self.N_s_a[state, action] += 1
                
                new_state, reward, terminated, truncated, _ = self.env.step(action)

                self.scores[i]+=reward

                steps.append((state, action, reward, terminated))

                state = new_state

            eps = 1 - i/self.episodes

            # optimize
            self.optimize(steps)


    def optimize(self, steps):
        Gt = 0
        for (state, action, reward, terminated) in reversed(steps):
            if terminated:
                self.Q_s_a[state, action] += (1/self.N_s_a[state, action])*(reward - self.Q_s_a[state, action])
                Gt = reward
            else:
                Gt = reward + self.gamma * Gt
                self.Q_s_a[state, action] += (1/self.N_s_a[state, action])*(Gt - self.Q_s_a[state, action])
        steps.clear()

    def plot_rewards(self):
        PATH = os.path.abspath(__file__)
        x = range(0, self.episodes)
        plt.plot(x, self.scores)
        plt.savefig(f"tabular_methods/MC/res/{self.mode}_{self.env.unwrapped.spec.id}.png")
    
    def test(self, env, n_episodes=5):
        for i in range (n_episodes):
            state = env.reset()[0]
            truncated = False
            terminated = False
            while not (truncated or terminated):
                action = self.Q_s_a[state].argmax()
                state, reward, terminated, truncated, _ = env.step(action)

env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode=None)
trainer = MC_trainer(env = env, gamma=0.9, episodes=10000, mode="every_encounter")
trainer.train()
trainer.plot_rewards()
env.close()
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="human")
trainer.test(env, n_episodes=1)
env.close()



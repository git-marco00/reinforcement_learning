import gymnasium as gym
from policy_network import Policy
import random
from tqdm import trange
import torch
from matplotlib import pyplot as plt
import os
# policy gradient RL

class PGO_trainer():
    def __init__(self, env, gamma, episodes, learning_rate = 0.01):
        self.env = env
        self.gamma = gamma
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.epsilon = 1

        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.scores = []

        self.policy_network = Policy(self.n_states, self.n_actions)
        self.optimizer = torch.optim.Adam(params = self.policy_network.parameters(), lr = self.learning_rate)
        
    def train(self):
        for ep in trange(self.episodes):
            truncated = False
            terminated = False
            state = self.env.reset()[0]
            self.scores.append(0)
            transitions = []

            while not (truncated or terminated):
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.policy_network(self.state_OHE(state)).argmax().item()

                new_state, reward, terminated, truncated, _ = self.env.step(action)

                transitions.append((state, action, reward))

                self.scores[ep]+=reward

                if terminated:
                    self.G_update(transitions)
                    self.optimize(transitions)
                
                self.scores[ep]+=reward
                
                state = new_state

                self.epsilon = 1 - ep/self.episodes


    def G_update(self, transitions):
        for i in reversed(range(0, len(transitions)-1)):
            state, action, reward = transitions[i]
            transitions[i]=(state, action, reward + self.gamma*transitions[i+1][2])

    def optimize(self, transitions):
        for (state, action, gt) in transitions:
            log_prob = torch.log(self.policy_network(self.state_OHE(state))[action])
            loss = -log_prob * gt
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def state_OHE(self, state):
        one_hot_encoding = torch.zeros(self.n_states)
        one_hot_encoding[state]=1
        return one_hot_encoding
    
    def plot_rewards(self):
        PATH = os.path.abspath(__file__)
        x = range(0, self.episodes)
        plt.plot(x, self.scores)
        plt.savefig(f"PGO/{self.env.unwrapped.spec.id}.png")
    
    def test(self, env, n_episodes=5):
        for i in range (n_episodes):
            state = env.reset()[0]
            truncated = False
            terminated = False
            while not (truncated or terminated):
                action = self.policy_network(self.state_OHE(state)).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)


            
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode=None)
trainer = PGO_trainer(env = env, gamma=0.9, episodes=1000)
trainer.train()
trainer.plot_rewards()
env.close()
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="human")
trainer.test(env)
env.close()

    

    

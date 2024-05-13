import gymnasium as gym
from networks.policy_network import Policy
from networks.Q_Network import Q_network
import random
from tqdm import trange
import torch
from matplotlib import pyplot as plt
import os
# policy gradient RL
# sicuro è sbagliato perchè non sto usando il replay buffer per l'allenamento di Q.
# se facessi quello sarebbe corretto ?

class PGO_trainer():
    def __init__(self, env, gamma, episodes, mode, learning_rate = 0.01):
        self.env = env
        self.gamma = gamma
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.epsilon = 1
        self.mode = mode

        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.scores = []

        self.policy_network = Policy(self.n_states, self.n_actions)
        self.optimizer = torch.optim.Adam(params = self.policy_network.parameters(), lr = 0.005)
        self.critic_network = Q_network(n_actions=self.n_actions, n_states=self.n_states)
        self.critic_optim = torch.optim.Adam(params = self.critic_network.parameters(), lr = self.learning_rate)
        
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

                transitions.append((state, action, new_state, reward, terminated))

                self.scores[ep]+=reward

                if terminated:
                    if self.mode == "actor_critic":
                        self.critic_optimizer(transitions)
                    self.G_update(transitions)
                    self.optimize(transitions)
                
                state = new_state

                self.epsilon = 1 - ep/self.episodes


    def G_update(self, transitions):
        for i in reversed(range(0, len(transitions)-1)):
            state, action, new_state, reward, terminated = transitions[i]
            transitions[i]=(state, action, new_state, reward + self.gamma*transitions[i+1][3], terminated)

    def critic_optimizer(self, transitions):
        current_q_list = []
        target_q_list = []

        for (state, action, new_state, reward, terminated) in transitions:
            if terminated:
                target = torch.tensor(reward)
            else:
                target = reward + self.gamma*self.critic_network(self.state_OHE(new_state)).max()
            
            current = self.critic_network(self.state_OHE(state))[action]

            current_q_list.append(current)
            target_q_list.append(target)

        loss = torch.nn.functional.mse_loss(torch.stack(current_q_list), torch.stack(target_q_list))

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()  

    def optimize(self, transitions):
        for (state, action, new_state, gt, terminated) in transitions:
            log_prob = torch.log(self.policy_network(self.state_OHE(state))[action])
            if self.mode == "monte_carlo":
                loss = -log_prob * gt
            elif self.mode == "actor_critic":
                loss = - log_prob * self.critic_network(self.state_OHE(state))[action]
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
        plt.savefig(f"PGO/res/{self.mode}_{self.env.unwrapped.spec.id}.png")
    
    def test(self, env, n_episodes=5):
        for i in range (n_episodes):
            state = env.reset()[0]
            truncated = False
            terminated = False
            while not (truncated or terminated):
                action = self.policy_network(self.state_OHE(state)).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)
     
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode=None)
trainer = PGO_trainer(env = env, gamma=0.9, episodes=3000, mode="actor_critic")
trainer.train()
trainer.plot_rewards()
env.close()
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="human")
trainer.test(env, n_episodes=1)
env.close()

    

    

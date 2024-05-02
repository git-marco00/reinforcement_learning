import gymnasium as gym
from Replay_buffer import ReplayBuffer
from Q_Network import Q_network
import random
from tqdm import trange
import torch
from matplotlib import pyplot as plt
import os

class DQN_trainer():
    def __init__(self, env, gamma, episodes, steps_2_converge, learning_rate = 0.01, batch_size = 32, max_len_buffer=1000):
        self.env = env
        self.gamma = gamma
        self.max_len_buffer = max_len_buffer
        self.episodes = episodes
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.steps_2_converge = steps_2_converge
        self.epsilon = 1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scores = []
        self.loss_mode = None

        self.replay_buffer = ReplayBuffer(maxlen=max_len_buffer)
        self.policy_Q = Q_network(n_actions=self.n_actions, n_states=self.n_states)
        self.target_Q = Q_network(n_actions=self.n_actions, n_states=self.n_states)
        self.target_Q.load_state_dict(self.policy_Q.state_dict())
        self.optimizer = torch.optim.Adam(params= self.policy_Q.parameters(), lr = self.learning_rate)

    def state_OHE(self, state):
        one_hot_encoding = torch.zeros(self.n_states)
        one_hot_encoding[state]=1
        return one_hot_encoding

    def train(self, mode="simple_DQN_loss"):
        self.loss_mode = mode
        step = 0
        is_rewarded = False
        for i in trange(self.episodes):
            truncated = False
            terminated = False
            state = self.env.reset()[0]
            self.scores.append(0)

            while not (terminated or truncated):
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.policy_Q(self.state_OHE(state)).argmax().item()
                
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                if reward == 1:
                    is_rewarded = True

                self.replay_buffer.append((state, action, new_state, reward, terminated))

                state = new_state

                if step % self.steps_2_converge == 0 and is_rewarded:
                    # converge target network to policy network 
                    self.target_Q.load_state_dict(self.policy_Q.state_dict())

                step += 1

                self.scores[i]+=reward
                
            # optimize policy network
            if len(self.replay_buffer) > self.batch_size and is_rewarded:
                self.optimize(mode)

            self.epsilon = 1 - i/self.episodes
            
    def optimize(self, mode):
        current_q_list = []
        target_q_list = []
        mini_batch = self.replay_buffer.sample(self.batch_size)

        for state, action, new_state, reward, terminated in mini_batch:
            
            if terminated:
                target = torch.tensor(reward)
            else:
                if mode == "simple_DQN_loss":
                    target = reward + self.gamma*self.target_Q(self.state_OHE(new_state)).max()
                elif mode == "double_DQN_loss":
                    action_ = self.policy_Q(self.state_OHE(new_state)).argmax()
                    target = reward + self.gamma*self.target_Q(self.state_OHE(new_state))[action_]
                    
            current = self.policy_Q(self.state_OHE(state))[action]

            current_q_list.append(current)
            target_q_list.append(target)

        loss = torch.nn.functional.mse_loss(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()     

    def test(self, env, n_episodes=5):
        for i in range (n_episodes):
            state = env.reset()[0]
            truncated = False
            terminated = False
            while not (truncated or terminated):
                action = self.policy_Q(self.state_OHE(state)).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)

    def plot_rewards(self):
        PATH = os.path.abspath(__file__)
        x = range(0, self.episodes)
        plt.plot(x, self.scores)
        plt.savefig(f"DQN/res/{self.env.unwrapped.spec.id}_{self.loss_mode}.png")


            
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode=None)
trainer = DQN_trainer(env = env, gamma=0.9, episodes=1000, steps_2_converge=30)
trainer.train(mode="simple_DQN_loss")
trainer.plot_rewards()
env.close()
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="human")
trainer.test(env)
env.close()
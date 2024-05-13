import gymnasium as gym
import random, math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from collections import deque
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

# STOCHASTIC POLICY WITH GAUSSIAN DISTRIBUTION

class Policy_Network(nn.Module):

	def __init__(self, in_size, out_size, hidden = 64):
		super().__init__()
		self.layer1 = nn.Linear(in_size, hidden)
		self.layer2 = nn.Linear(hidden, hidden)
		self.layer3 = nn.Linear(hidden, out_size*2)

		self.out_size = out_size
		
		self.scale_factor = torch.tensor([1.5, 2], requires_grad = False)
		self.bias_factor = torch.tensor([- 0.5, -1], requires_grad = False)
        
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = F.sigmoid(self.layer3(x))	# [0,1] => [0, 1.5] => [-0.5, 1]


		return x[:, :self.out_size] * self.scale_factor + self.bias_factor, x[:, self.out_size:] + 0.01
		
	def __str__(self):
		return "PG"
		
		
class PGA_Reinforce():

	def __init__(self, env, continuous, episodes, gamma, learning_rate, network, traj_batch_size, early_stopping_window, threshold):
		self.env_name = env
		self.env = gym.make(self.env_name, continuous = continuous)
		self.episodes = episodes
		self.gamma = gamma
		
		self.score = []
		self.var_main = []
		self.var_sec = []
		self.var_main_avg = []
		self.var_sec_avg = []
		
		self.traj_batch_size = traj_batch_size
		self.early_stopping_window = math.ceil(early_stopping_window/traj_batch_size)
		self.threshold = threshold
		self.continuous = continuous
		
		self.action_size = self.env.action_space.shape[0]
		self.state_size = self.env.observation_space.shape[0]
		
		self.learning_rate = learning_rate
		
		self.memory_state = []
		self.memory_action = []
		self.memory_reward = []
		
		self.model = network(self.state_size, self.action_size)
		self.model.eval()
		
		self.criterion = nn.SmoothL1Loss()
		self.optimizer = optim.AdamW(self.model.parameters(), lr = learning_rate, amsgrad=True)
		
		self.I = torch.eye(self.action_size)
		
		
	def reset_agent(self):
		self.memory_state = []
		self.memory_action = []
		self.memory_reward = []
		
			
	def add_experience(self, state, action, reward):
		self.memory_state[0].insert(0, state)
		self.memory_action[0].insert(0, action)
		self.memory_reward[0].insert(0, reward)


	def select_action(self, state):
		with torch.no_grad():
			means, stds = self.model(state.unsqueeze(0))
			self.var_main.append(stds[:, 0])
			self.var_sec.append(stds[:, 1])
			m = MultivariateNormal(means, self.I*stds)
			a = m.sample().squeeze()
			return a
	
	
	def experience_replay(self):
		
		for i in np.arange(self.traj_batch_size):
			current_return = torch.tensor(0.0)
			self.memory_state[i] = torch.stack(self.memory_state[i])
			self.memory_action[i] = torch.stack(self.memory_action[i])
			self.memory_reward[i] = torch.stack(self.memory_reward[i])
			
			for j in np.arange(self.memory_reward[i].shape[0]):
				self.memory_reward[i][j] += self.gamma * current_return
				current_return = self.memory_reward[i][j]
					
		
		state_batch = torch.vstack(self.memory_state)
		action_batch = torch.vstack(self.memory_action)
		return_batch = torch.hstack(self.memory_reward)
			

		self.model.train()
		
		means, stds = self.model(state_batch)
		m = MultivariateNormal(means, torch.einsum('ij, jh -> ijh', stds, self.I))
		log_probs = m.log_prob(action_batch)

		loss = -log_probs * return_batch
		
		self.optimizer.zero_grad()
		loss.mean().backward()
		self.optimizer.step()
		
		self.model.eval()
		

	def learn(self):
		for e in tqdm(np.arange(episodes), desc="Learning"):
		
			score = 0.0
			
			if len(self.score) >= self.early_stopping_window and np.mean(self.score[-self.early_stopping_window:]) >= self.threshold:
				break
				
			if len(self.score) >= self.early_stopping_window and np.mean(self.score[-self.early_stopping_window:]) >= 150:
				self.optimizer.param_groups[0]['lr'] = 1e-3
		
			for t in np.arange(self.traj_batch_size):
				
				state, _ = self.env.reset()
				state[5] *= 2.5
				state = torch.tensor(state)
				episode_score = 0.0
				done = False
				self.memory_state.insert(0, [])
				self.memory_action.insert(0, [])
				self.memory_reward.insert(0, [])
							
				while not done:
					action = self.select_action(state)
					next_state, reward, terminated, truncated, _ = self.env.step(action.numpy())
					
					done = terminated or truncated
					
					next_state[5] *= 2.5
					next_state = torch.tensor(next_state)

					episode_score += reward
					reward = torch.tensor(reward)
					
					self.add_experience(state, action, reward)
					
					state = next_state	
			
				score += (episode_score - score)/(t+1)

			self.var_main = [tensor.item() for tensor in self.var_main]
			self.var_sec = [tensor.item() for tensor in self.var_sec]
			
			self.var_main_avg.append(np.mean(self.var_main))
			self.var_sec_avg.append(np.mean(self.var_sec))
			self.var_main = []
			self.var_sec = []
			
			self.score.append(score)
			self.experience_replay()	
			self.reset_agent()
					
		self.env.close()
		
		
	def simulate(self):
		env = gym.make(self.env_name, continuous = self.continuous, render_mode = "human")
		done = False
		state, _ = env.reset()
		state = torch.tensor(state)
		while not done:
			action = self.select_action(state)
			next_state, _, terminated, truncated, _ = env.step(action.numpy())

			done = terminated or truncated

			state = torch.tensor(next_state)

		env.close()


	def plot_learning(self, N, title=None, filename = ""):
		time = len(self.score)
		start = math.floor(N/2)
		end = time-start
		plt.plot(self.score)
		mean_score = np.convolve(np.array(self.score), np.ones(N)/N, mode='valid')
		plt.plot(np.arange(start,end), mean_score)

		if title is not None:
			plt.title(title)

		plt.savefig(self.env_name+filename)
		plt.clf()
		
	def plot_variance(self, title=None, filename = ""):
		time = len(self.var_main_avg)
		plt.plot(self.var_main_avg, label="main")
		plt.plot(self.var_sec_avg, label="secondary")
		plt.legend()

		if title is not None:
			plt.title(title)

		plt.savefig(self.env_name+filename)
		plt.clf()
		
	
	def save_model(self, path=""):
		torch.save(self.model.state_dict(), path+self.env_name+"_"+str(self)+".pt")

	
	def load_model(self, path):
		self.model.load_state_dict(torch.load(path+self.env_name+"_"+str(self)+".pt"))
		self.model.eval()
		
	
	def __str__(self):
		return "PGA_Reinforce"


if __name__ == '__main__':
	num_cores = 8
	torch.set_num_interop_threads(num_cores) # Inter-op parallelism
	torch.set_num_threads(num_cores) # Intra-op parallelism
	
	torch.autograd.set_detect_anomaly(True)
	
	env = 'LunarLander-v2'
	network = Policy_Network
	episodes = 400
	gamma = 0.99
	learning_rate = 3e-3
	early_stopping_window = 75
	threshold = 200
	continuous = True
	traj_batch_size = 15
	
	agent = PGA_Reinforce(env, continuous, episodes, gamma, learning_rate, network, traj_batch_size, early_stopping_window, threshold)
	
	agent.learn()
	agent.plot_learning(31, title = "Lunar Lander", filename = str(agent))
	agent.plot_variance(title = "Lunar Lander variance", filename = str(agent)+"_variance")
	
	agent.simulate()
	
	agent.save_model()






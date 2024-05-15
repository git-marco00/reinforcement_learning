import gymnasium as gym
import random, math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from collections import deque
from abc import ABC
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Q_Network(nn.Module):

	def __init__(self, n_observations, n_actions, hidden = 64):
		super().__init__()
		self.layer1 = nn.Linear(n_observations, hidden)
		self.layer2 = nn.Linear(hidden, hidden)
		self.layer3 = nn.Linear(hidden, n_actions)
        
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		return self.layer3(x)


class Dueling_Network(nn.Module):

	def __init__(self, n_observations, n_actions, hidden = 64):
		super().__init__()
		self.layer1 = nn.Linear(n_observations, hidden)
		self.layer2 = nn.Linear(hidden, hidden)

		self.V = nn.Linear(hidden, 1)

		self.A = nn.Linear(hidden, n_actions)
		
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		V = self.V(x)
		A = self.A(x)

		return V + (A - A.mean(dim=1, keepdim=True))


class DeepAgent(ABC):

	def __init__(self, env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, learning_rate, batch_size, train_start, 
					replay_frequency, target_model_update_rate, memory_length, mini_batches, agent_type, early_stopping_window, threshold, mode = "double_DQN"):
		self.env_name = env
		self.env = gym.make(self.env_name)
		self.epsilon = epsilon_start
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.episodes = episodes
		self.gamma = gamma
		self.replay_frequency = replay_frequency
		#self.target_model_update_rate = target_model_update_rate
		self.mini_batches = mini_batches
		self.score = []
		self.epsilon_record = []
		self.early_stopping_window = early_stopping_window
		self.threshold = threshold
		self.loss_history = []
		self.mode = mode
		
		self.action_space = self.env.action_space
		self.action_size = self.env.action_space.n
		self.state_size = self.env.observation_space.shape[0]
		
		
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.train_start = train_start
		
		self.memory = deque(maxlen=memory_length)
		
		self.agent_type = agent_type
		
		if self.agent_type == "DDDQN":
			self.model = Dueling_Network(self.state_size, self.action_size)
			self.target_model = Dueling_Network(self.state_size, self.action_size)
		else:
			self.model = Q_Network(self.state_size, self.action_size)
			self.target_model = Q_Network(self.state_size, self.action_size)

		self.target_model.load_state_dict(self.model.state_dict())
		self.model.eval()
		self.target_model.eval()
		
		self.criterion = nn.SmoothL1Loss()
		self.optimizer = optim.AdamW(self.model.parameters(), lr = learning_rate, amsgrad=True)
		
		
	def update_epsilon(self):
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
		
			
	def add_experience(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))


	def select_action(self, state):
		if random.random() <= self.epsilon:
			return self.action_space.sample()
		
		with torch.no_grad():
			return torch.argmax(self.model(state.unsqueeze(0))).item()

    
	def update_target_model(self, tau):
		weights = self.model.state_dict()
		target_weights = self.target_model.state_dict()
		for i in target_weights:
			target_weights[i] = weights[i] * tau + target_weights[i] * (1-tau)
		self.target_model.load_state_dict(target_weights)
	
	
	def q_value_arrival_state(self, states):
		pass
	
	
	def experience_replay(self):
			"""
			
			if len(self.memory) < self.train_start:
				return
	
			batch = random.sample(self.memory, self.batch_size)
			
			actions = [torch.tensor(x[1]) for x in batch]
			states = [x[0] for x in batch]
			next_states = [x[3] for x in batch]
			rewards = [x[2] for x in batch]
			terminated = [torch.tensor(x[4]) for x in batch]
			
			actions_batch = torch.stack(actions)
			states_batch = torch.stack(states)
			next_states_batch = torch.stack(next_states)
			rewards_batch = torch.stack(rewards)
			terminated_batch = torch.stack(terminated)

			# Q(s,a,w) = reward + gamma * max a' Q(s', a', w-) - Q(s, a, w)
			# questo perchè io voglio che la mia Q (ovvero l'actor) abbia un valore che si avvicini il più possibile
			# alla reward che ho ottenuto effettivamente facendo quell'azione + tutto ciò che ottengo dopo
			# ma utilizzando una rete stabile ovver Q(w-)

			# current batch
			q_batch = self.model(states_batch)
			current_batch = torch.gather(q_batch, 1, actions_batch.unsqueeze(dim=1)).squeeze() # Q(s, a, w)
			
			# target
			if self.mode == "simple_DQN":
				with torch.no_grad():
					next_q_batch = self.target_model(next_states_batch)
					next_max_q_batch,_ = torch.max(next_q_batch, dim=1)
					target_batch = rewards_batch + ~terminated_batch * self.gamma * next_max_q_batch    # reward + gamma * max a' Q(s', a', w-)
			
			elif self.mode == "double_DQN":
				with torch.no_grad():
					next_q_batch_theta_meno = self.model(next_states_batch)
					best_actions_batch = torch.argmax(next_q_batch_theta_meno, dim=1).unsqueeze(1)
					next_q_batch_theta = self.target_model(next_states_batch)
					next_q_batch = torch.gather(next_q_batch_theta, 1, best_actions_batch).squeeze()
					target_batch = rewards_batch + ~terminated_batch * self.gamma * next_q_batch
					
			loss = self.criterion(current_batch, target_batch)
			self.optimizer.zero_grad()
			self.loss_history.append(loss.item())
			loss.backward()
			self.optimizer.step()	

			"""

			if len(self.memory) < self.train_start:
				return
			
			batch = random.sample(self.memory, self.batch_size)
			state_batch = torch.empty((self.batch_size, self.state_size))
			next_state_batch = torch.empty((self.batch_size, self.state_size))
			action_batch = torch.empty((self.batch_size, 1), dtype = torch.long)
			reward_batch = torch.empty((self.batch_size,1))
			not_done_batch = torch.empty(self.batch_size, dtype = torch.bool)
			
			for i in np.arange(self.batch_size):
				state_batch[i,:] = batch[i][0]
				action_batch[i,0] = batch[i][1]
				reward_batch[i,0] = batch[i][2]
				next_state_batch[i,:] = batch[i][3]
				not_done_batch[i] = not batch[i][4]
				
			next_state_values = torch.zeros((self.batch_size,1))
			
			with torch.no_grad():
				next_state_values[not_done_batch] = self.q_value_arrival_state(next_state_batch[not_done_batch,:])
				target_values = reward_batch + self.gamma * next_state_values

			self.model.train()
			predicted_values = self.model(state_batch).gather(1,action_batch)

			loss = self.criterion(predicted_values, target_values)
			
			self.optimizer.zero_grad()
			self.loss_history.append(loss.item())
			loss.backward()
			self.optimizer.step()
			
			self.model.eval()
				
			
			
	def learn(self):
		for e in tqdm(np.arange(episodes), desc="Learning"):
		
			if len(self.score) >= self.early_stopping_window and np.mean(self.score[-self.early_stopping_window:]) >= self.threshold:
				break
			
			state, _ = self.env.reset()
			state[5] *= 2.5
			state = torch.tensor(state)
			episode_score = 0
			step = 0
			done = False
						
			while not done:
				action = self.select_action(state)
				next_state, reward, terminated, truncated, _ = self.env.step(action)
				
				done = terminated or truncated
				
				next_state[5] *= 2.5
				next_state = torch.tensor(next_state)

				episode_score += reward
				reward = torch.tensor(reward)
				
				self.add_experience(state, action, reward, next_state, terminated)
				
				state = next_state

				step += 1
				
				if (step & self.replay_frequency) == 0:
					for i in np.arange(mini_batches):
						self.experience_replay()
						self.target_model.load_state_dict(self.model.state_dict())

				if done:								
					self.update_epsilon()
					self.score.append(episode_score)
					self.epsilon_record.append(self.epsilon)
					
		self.env.close()
		
		
	def simulate(self):
		env = gym.make(self.env_name, render_mode = "human")
		self.epsilon = -1
		done = False
		state, _ = env.reset()
		state = torch.tensor(state)
		while not done:
			action = self.select_action(state)
			next_state, _, terminated, truncated, _ = env.step(action)

			done = terminated or truncated

			state = torch.tensor(next_state)

		env.close()


	def plot_learning(self, N, title=None, filename = ""):
		time = len(self.score)
		start = math.floor(N/2)
		end = time-start
		plt.plot(self.score);
		mean_score = np.convolve(np.array(self.score), np.ones(N)/N, mode='valid')
		plt.plot(np.arange(start,end), mean_score)

		if title is not None:
			plt.title(title);

		plt.savefig(filename)
		plt.clf()


	def plot_epsilon(self):
		plt.plot(self.epsilon_record);
		plt.title("Epsilon decay");
		plt.savefig(self.env_name + "_epsilon")
		plt.clf()
		
	
	def save_model(self, path=""):
		torch.save(self.model.state_dict(), path+self.env_name+"_"+self.agent_type+".pt")

	
	def load_model(self, path):
		self.model.load_state_dict(torch.load(path+self.env_name+"_"+self.agent_type+".pt"))
		self.model.eval()
		


class DQNA(DeepAgent):
	
	def __init__(self, env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, learning_rate, batch_size, train_start, 
					replay_frequency, target_model_update_rate, memory_length, mini_batches, agent_type, early_stopping_window, threshold, mode):
		super().__init__(env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, learning_rate, batch_size, train_start,
						 replay_frequency, target_model_update_rate, memory_length, mini_batches, agent_type, early_stopping_window, threshold, mode)
	
	
	def q_value_arrival_state(self, states):
		actions = torch.argmax(self.model(states), dim = 1).unsqueeze(1)
		return self.target_model(states).gather(1, actions)

	

if __name__ == '__main__':
	num_cores = 8
	torch.set_num_interop_threads(num_cores) # Inter-op parallelism
	torch.set_num_threads(num_cores) # Intra-op parallelism
	
	env = 'LunarLander-v2'
	#agent_type = "DDDQN"
	agent_type = "DQN"
	episodes = 100
	#episodes = 400
	replay_frequency = 3
	#replay_frequency = 7
	gamma = 0.99
	#learning_rate = 5e-4
	learning_rate = 1e-3
	epsilon_start = 1
	epsilon_decay = 0.992
	epsilon_min = 0.01
	batch_size = 64
	train_start = 128
	target_model_update_rate = 1e-2
	#target_model_update_rate = 5e-3
	memory_length = 100000
	mini_batches = 2
	#early_stopping_window = 20
	early_stopping_window = 100
	#threshold = 200
	threshold = 220
	
	agent = DQNA(env, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, learning_rate, batch_size, train_start,
				replay_frequency, target_model_update_rate, memory_length, mini_batches, agent_type, early_stopping_window, threshold,
				mode = "simple_DQN")
	

	score_file = os.getcwd()+"\DQN\\res\\DDQN_score_Fiaschi.png"
	loss_file = os.getcwd()+"\DQN\\res\\DDQN_loss_Fiaschi.png"

	agent.learn()
	
	agent.plot_learning(31, title = "Lunar Lander", filename = score_file)

	window_size = 100
	smoothed_data = np.convolve(agent.loss_history, np.ones(window_size)/window_size, mode='valid')
	plt.plot(smoothed_data)
	plt.savefig(loss_file)
	plt.clf()
	
	#agent.simulate()
	
	#agent.save_model()

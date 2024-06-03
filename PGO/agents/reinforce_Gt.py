import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import gymnasium as gym
import matplotlib.pyplot as plt
from networks.stochastic_network import Actor as PolicyNetwork
from tqdm import trange
import torch
from torch.distributions.categorical import Categorical
import numpy as np
from collections import deque
from statistics import mean
import time
from agent_interface import Agent
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Reinforce_agent(Agent):
	def __init__(self, env, lr, n_trajectories, n_episodes, gamma, solved_reward, early_stopping_window, model_path):
		self.env = env
		self.n_actions = 2
		self.n_states = 4
		self.lr = lr
		self.n_trajectories = n_trajectories
		self.n_episodes = n_episodes
		self.gamma = gamma
		self.model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), model_path))

		self.policy_network = PolicyNetwork(n_states=self.n_states, n_actions=self.n_actions)
		self.policy_opt = torch.optim.Adam(params = self.policy_network.parameters(), lr = self.lr)

		self.scores = []
		self.loss_history=[]

		self.state_buffer = []
		self.action_buffer = []
		self.reward_buffer = []
		self.logprob_buffer = []

		self.solved_reward = solved_reward
		self.early_stopping_window = early_stopping_window
		self.window = deque(maxlen=self.early_stopping_window)
		
		self.loss_fn = torch.nn.MSELoss()

	def train(self):
		start_timestamp = time.time()
		t = trange(self.n_episodes)
		for ep in t:
			episode_scores = []
			for traj in range(self.n_trajectories):

				self.state_buffer.append([])
				self.action_buffer.append([])
				self.reward_buffer.append([])
				self.logprob_buffer.append([])

				state = self.env.reset()[0]
				state = torch.tensor(state)

				truncated = False
				terminated = False
				episode_scores.append(0)
				while not (truncated or terminated):
					distr_params = self.policy_network(state.unsqueeze(0))

					distr = Categorical(distr_params.squeeze())
					
					action = distr.sample()	# returns a tensor
					
					log_prob = distr.log_prob(action)

					new_state, reward, terminated, truncated, _ = self.env.step(action.numpy())	# tensor => numpy array

					episode_scores[-1]+=reward

					self.state_buffer[traj].append(state)
					self.action_buffer[traj].append(action)
					self.reward_buffer[traj].append(reward)
					self.logprob_buffer[traj].append(log_prob)

					state = torch.tensor(new_state)

				# early stopping condition
				self.window.append(episode_scores[-1])
				if int(mean(self.window)) == self.solved_reward:
					end_timestamp = time.time()
					total_time = end_timestamp - start_timestamp
					print(f"[ENVIRONMENT SOLVED in {total_time} seconds]")
					# save model
					self.save_model(self.model_path)
					t.close()
					return
				
			
			self.scores.append(np.mean(episode_scores))
			t.set_description(f"score: {round(np.mean(episode_scores), 2)}")
			
			episode_scores.clear()

			self.compute_returns()

			self.optimize()

			self.state_buffer.clear()
			self.action_buffer.clear()
			self.reward_buffer.clear()
			self.logprob_buffer.clear()
		
		# EPISODE NOT SOLVED
		end_timestamp = time.time()
		total_time = end_timestamp - start_timestamp
		print(f"[ENVIRONMENT NOT SOLVED. Elapsed time: {total_time} seconds]")
		self.save_model(self.model_path)

	def save_model(self, path):
		torch.save(self.policy_network.state_dict(), path)

	def load_model(self, path):
		self.policy_network = PolicyNetwork(self.n_states, self.n_actions)
		self.policy_network.load_state_dict(torch.load(path))

				
	def compute_returns(self):
		for traj in self.reward_buffer:
			for i in reversed(range(len(traj)-1)):
				traj[i] = traj[i] + self.gamma*traj[i+1]
				traj[i] = torch.Tensor([traj[i]])
			traj[len(traj)-1]= torch.Tensor([traj[len(traj)-1]])

		
	def optimize(self):
		# self.state_buffer = [15, 104, 8]
		for i in range(self.n_trajectories):
			self.reward_buffer[i] = torch.stack(self.reward_buffer[i])
			self.logprob_buffer[i] = torch.stack(self.logprob_buffer[i])

		reward_batch = torch.cat(self.reward_buffer, 0).squeeze()
		logprobs_batch = torch.cat(self.logprob_buffer, 0)

		norm_reward = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-6)
		
		loss = -logprobs_batch * norm_reward	# torch.Size([1589])

		self.policy_opt.zero_grad()
		self.loss_history.append(loss.mean().item())
		loss.mean().backward()
		self.policy_opt.step()
				

	def plot(self):
		PATH = os.path.abspath(__file__)
		plt.plot(self.scores)
		plt.title("Score")
		plt.xlabel("Episode")
		plt.ylabel("Reward")
		plt.savefig("reinforce_Gt_score.png")
		plt.clf()

		plt.plot(self.loss_history)
		plt.title("Actor loss")
		plt.xlabel("Episode")
		plt.ylabel("Loss")
		plt.savefig("reinforce_Gt_actor_loss.png")
		plt.clf()

	def test(self, env, n_episodes=5):
		self.load_model(self.model_path)
		for i in range (n_episodes):
			truncated = False
			terminated = False
			state = env.reset()[0]
			state = torch.Tensor(state)
			while not (truncated or terminated):
				distr_params= self.policy_network(state.unsqueeze(0))

				distr = Categorical(distr_params.squeeze())
					
				action = distr.sample()	# returns a tensor

				new_state, reward, terminated, truncated, _ = env.step(action.numpy())	# tensor => numpy array

				state = torch.tensor(new_state)

def main():				
	num_cores = 8
	torch.set_num_interop_threads(num_cores) # Inter-op parallelism
	torch.set_num_threads(num_cores) # Intra-op parallelism
	env = gym.make('CartPole-v1', render_mode=None)

	solved_reward = 500
	early_stopping_window = 20
	lr=0.001
	n_trajectories=10
	n_episodes=500
	gamma=0.99

	model_path = "saved_models\\REINFORCE_Gt_policy_2"
	trainer = Reinforce_agent(env=env, lr=lr, n_trajectories=n_trajectories, n_episodes=n_episodes, gamma=gamma, solved_reward=solved_reward, early_stopping_window=early_stopping_window, model_path = model_path)
	trainer.train()
	trainer.plot()
	env.close()
	env = gym.make('CartPole-v1', render_mode = "human")
	trainer.test(n_episodes=5, env=env)
	env.close()

if __name__ == '__main__':
    main()  
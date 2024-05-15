import gymnasium as gym
import matplotlib.pyplot as plt
import os
from networks.lunar_network import PolicyNetwork
from tqdm import trange
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

class Reinforce():
	def __init__(self, env, lr, n_trajectories, n_episodes, gamma):
		self.env = env
		self.n_actions = self.env.action_space.shape[0]
		self.n_states = self.env.observation_space.shape[0]
		self.lr = lr
		self.n_trajectories = n_trajectories
		self.n_episodes = n_episodes
		self.gamma = gamma

		self.policy_network = PolicyNetwork(n_states=self.n_states, n_actions=self.n_actions)
		self.optimizer = torch.optim.Adam(params = self.policy_network.parameters(), lr = self.lr)

		self.scores = []
		self.loss_history=[]

		self.state_buffer = []
		self.action_buffer = []
		self.reward_buffer = []

	def train(self):
		for ep in trange(self.n_episodes):
			episode_scores = []
			for traj in range(self.n_trajectories):

				self.state_buffer.append([])
				self.action_buffer.append([])
				self.reward_buffer.append([])

				state = self.env.reset()[0]
				state[5] *= 2.5
				state = torch.tensor(state)

				
				truncated = False
				terminated = False
				episode_scores.append(0)
				while not (truncated or terminated):
					means, stds = self.policy_network(state.unsqueeze(0))

					means = means.squeeze()
					stds = stds.squeeze()

					distr = MultivariateNormal(means, torch.diag(stds))
					
					action = distr.sample()	# returns a tensor

					new_state, reward, terminated, truncated, _ = self.env.step(action.numpy())	# tensor => numpy array

					episode_scores[-1]+=reward

					self.state_buffer[traj].append(state)
					self.action_buffer[traj].append(action)
					self.reward_buffer[traj].append(reward)

					new_state[5] *= 2.5

					state = torch.tensor(new_state)
				
			
			self.scores.append(np.mean(episode_scores))
			episode_scores.clear()

			self.compute_returns()

			self.optimize()

			self.state_buffer.clear()
			self.action_buffer.clear()
			self.reward_buffer.clear()

				
	def compute_returns(self):
		for traj in self.reward_buffer:
			for i in reversed(range(len(traj)-1)):
				traj[i] = traj[i] + self.gamma*traj[i+1]
				traj[i] = torch.Tensor([traj[i]])
			traj[len(traj)-1]= torch.Tensor([traj[len(traj)-1]])

		
		
	def optimize(self):
		# self.state_buffer = [15, 104, 8]
		for i in range(self.n_trajectories):
			self.state_buffer[i] = torch.stack(self.state_buffer[i])	#  => Torch.size([104, 8])
			self.action_buffer[i] = torch.stack(self.action_buffer[i])
			self.reward_buffer[i] = torch.stack(self.reward_buffer[i])

		state_batch = torch.cat(self.state_buffer, 0)		# => Torch.size([1858, 8])
		action_batch = torch.cat(self.action_buffer, 0)
		reward_batch = torch.cat(self.reward_buffer, 0).squeeze()

		means, stds = self.policy_network(state_batch)
		stds_diags = torch.Tensor(stds.shape[0], 2, 2)
		for i in range(stds.shape[0]):
			stds_diags[i] = torch.diag(stds[i])
		
		m = MultivariateNormal(means, stds_diags)
		
		log_probs = m.log_prob(action_batch)	

		loss = -log_probs * reward_batch	# torch.Size([1589])

		self.optimizer.zero_grad()
		self.loss_history.append(loss.mean())
		loss.mean().backward()
		self.optimizer.step()
				

	def plot_rewards(self):
		PATH = os.path.abspath(__file__)
		x = range(0, self.n_episodes)
		plt.plot(x, self.scores)
		plt.savefig(f"PGO/res/my_SPGD_score.png")
		plt.clf()

		plt.plot(self.loss_history)
		plt.savefig(f"PGO/res/my_SPGD_loss.png")

	def test(self, env, n_episodes=5):
		for i in range (n_episodes):
			truncated = False
			terminated = False
			state = env.reset()[0]
			state[5] *= 2.5
			state = torch.Tensor(state)
			while not (truncated or terminated):
				means, stds = self.policy_network(state.unsqueeze(0))

				means = means.squeeze()
				stds = stds.squeeze()

				distr = MultivariateNormal(means, torch.diag(stds))
					
				action = distr.sample()	# returns a tensor

				new_state, reward, terminated, truncated, _ = env.step(action.numpy())	# tensor => numpy array

				new_state[5] *= 2.5

				state = torch.tensor(new_state)
						
num_cores = 8
torch.set_num_interop_threads(num_cores) # Inter-op parallelism
torch.set_num_threads(num_cores) # Intra-op parallelism
env = gym.make('LunarLander-v2', continuous = True, render_mode=None)
trainer = Reinforce(env=env, lr=3e-3, n_trajectories=15, n_episodes=400, gamma=0.99)
trainer.train()
trainer.plot_rewards()
env.close()
env = gym.make('LunarLander-v2', continuous = True, render_mode="human")
trainer.test(env, n_episodes=5)
env.close()







				



					
		

                
        

                
			






import gymnasium as gym
import matplotlib.pyplot as plt
import os
from networks.stochastic_network import Actor as PolicyNetwork
from networks.stochastic_network import Critic
from tqdm import trange
import torch
from torch.distributions.categorical import Categorical
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Reinforce():
	def __init__(self, env, lr, n_trajectories, n_episodes, gamma):
		self.env = env
		self.n_actions = 2
		self.n_states = 4
		self.lr = lr
		self.n_trajectories = n_trajectories
		self.n_episodes = n_episodes
		self.gamma = gamma

		self.policy_network = PolicyNetwork(n_states=self.n_states, n_actions=self.n_actions)
		self.policy_opt = torch.optim.Adam(params = self.policy_network.parameters(), lr = self.lr)
		self.critic_network = Critic(input_dim=self.n_states, output_dim=1)
		self.critic_opt = torch.optim.Adam(params = self.critic_network.parameters(), lr = self.lr)

		self.scores = []
		self.critic_loss_history = []
		self.policy_loss_history = []

		self.state_buffer = []
		self.action_buffer = []
		self.reward_buffer = []
		
		self.loss_fn = torch.nn.MSELoss()

	def train(self):
		t = trange(self.n_episodes)
		for ep in t:
			episode_scores = []
			for traj in range(self.n_trajectories):

				self.state_buffer.append([])
				self.action_buffer.append([])
				self.reward_buffer.append([])

				state = self.env.reset()[0]
				state = torch.tensor(state)

				truncated = False
				terminated = False
				episode_scores.append(0)
				while not (truncated or terminated):
					distr_params = self.policy_network(state.unsqueeze(0))

					distr = Categorical(distr_params.squeeze())
					
					action = distr.sample()	# returns a tensor

					new_state, reward, terminated, truncated, _ = self.env.step(action.numpy())	# tensor => numpy array

					episode_scores[-1]+=reward

					self.state_buffer[traj].append(state)
					self.action_buffer[traj].append(action)
					self.reward_buffer[traj].append(reward)

					state = torch.tensor(new_state)
				
			
			self.scores.append(np.mean(episode_scores))
			t.set_description(f"score: {round(np.mean(episode_scores), 2)}")
			
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
		
        ## critic
		values = self.critic_network(state_batch)
		loss = self.loss_fn(values.squeeze(), reward_batch)
		self.critic_loss_history.append(loss.item())
		self.critic_opt.zero_grad()
		loss.backward()
		self.critic_opt.step()
		
		with torch.no_grad():
			values = self.critic_network(state_batch).squeeze()

		distr_params = self.policy_network(state_batch)
		m = Categorical(distr_params)
		advantages = reward_batch - values
		
		log_probs = m.log_prob(action_batch)	

		loss = -log_probs * advantages	# torch.Size([1589])

		self.policy_opt.zero_grad()
		self.policy_loss_history.append(loss.mean().item())
		loss.mean().backward()
		self.policy_opt.step()
				

	def plot_rewards(self):
		PATH = os.path.abspath(__file__)
		plt.plot(self.scores)
		plt.savefig(f"PGO/res/REINFORCE/reinforce_ADVANTAGE_score.png")
		plt.clf()

		plt.plot(self.critic_loss_history)
		plt.savefig(f"PGO/res/REINFORCE/reinforce_ADVANTAGE_critic_loss.png")
		plt.clf()

		plt.plot(self.policy_loss_history)
		plt.savefig(f"PGO/res/REINFORCE/reinforce_ADVANTAGE_policy_loss.png")
		plt.clf()

	def test(self, env, n_episodes=5):
		for i in range (n_episodes):
			truncated = False
			terminated = False
			state = env.reset()[0]
			state = torch.Tensor(state)
			while not (truncated or terminated):
				distr_params= self.policy_network(state.unsqueeze(0))

				distr = Categorical(distr_params)
					
				action = distr.sample()	# returns a tensor

				new_state, reward, terminated, truncated, _ = env.step(action.numpy())	# tensor => numpy array

				state = torch.tensor(new_state)
						
num_cores = 8
torch.set_num_interop_threads(num_cores) # Inter-op parallelism
torch.set_num_threads(num_cores) # Intra-op parallelism
env = gym.make('CartPole-v1', render_mode=None)
trainer = Reinforce(env=env, lr=0.001, n_trajectories=10, n_episodes=200, gamma=0.99)
trainer.train()
trainer.plot_rewards()
env.close()

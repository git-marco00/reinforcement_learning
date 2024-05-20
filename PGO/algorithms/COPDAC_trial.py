import gymnasium as gym
import matplotlib.pyplot as plt
import os
from networks.deterministic_network import Compatible_Deterministic_Q, Deterministic_Policy
from tqdm import trange
import torch
import numpy as np
from my_utils.Replay_buffer import ReplayBuffer
import random

class Reinforce():
	def __init__(self, env, Q_lr, policy_lr, n_episodes, gamma, buffer_max_len, steps2opt, batch_size):
		self.env = env
		self.n_actions = self.env.action_space.shape[0]
		self.n_states = self.env.observation_space.shape[0]
		self.Q_lr = Q_lr
		self.policy_lr = policy_lr
		self.n_episodes = n_episodes
		self.gamma = gamma
		self.buffer_max_len = buffer_max_len
		self.steps2opt = steps2opt
		self.batch_size = batch_size

		self.replay_buffer = ReplayBuffer(maxlen=buffer_max_len)
		self.Q = Compatible_Deterministic_Q(input_dim = self.n_actions)
		self.policy = Deterministic_Policy(n_states=self.n_states, n_actions=self.n_actions)

		self.Q_optim = torch.optim.Adam(params = self.Q.parameters(), lr = self.Q_lr)
		self.policy_optim = torch.optim.Adam(params = self.policy.parameters(), lr = self.policy_lr)
		self.loss = torch.nn.SmoothL1Loss()
		
		self.scores = []
		self.Q_loss_history=[]
		self.policy_loss_history=[]

	def train(self):
		step = 0
		t = trange(self.n_episodes)
		for ep in t:
			state = self.env.reset()[0]
			state = torch.tensor(state)

			truncated = False
			terminated = False

			self.scores.append(0)

			while not (truncated or terminated):
				actions = self.policy(state.unsqueeze(0)).detach().squeeze()

				new_state, reward, terminated, truncated, _ = self.env.step(actions.numpy())    # tensor => numpy array
				
				self.replay_buffer.append((state, actions, new_state, reward, terminated, truncated))
				step += 1
				self.scores[-1]+=reward

				state = torch.tensor(new_state)

				if step % self.steps2opt == 0 and len(self.replay_buffer) > self.batch_size*2:
					self.optimize()
				
			t.set_description(f"Episode score: {round(self.scores[-1], 2)}")
		
	def optimize(self):
		batch = self.replay_buffer.sample(sample_size=self.batch_size)

		states = [x[0] for x in batch]
		actions = [x[1] for x in batch]
		next_states = [torch.tensor(x[2]) for x in batch]
		rewards = [torch.tensor(x[3], dtype=torch.float32) for x in batch]
		terminated = [torch.tensor(x[4]) for x in batch]
		
		actions_batch = torch.stack(actions)
		states_batch = torch.stack(states)
		next_states_batch = torch.stack(next_states)
		rewards_batch = torch.stack(rewards)
		terminated_batch = torch.stack(terminated)

		state_action_batch = torch.cat((states_batch, actions_batch), dim=1)
		
		# ################ POLICY OPTIMIZATION ####################
		q_pred = 
		
		
		################## BACKWARDS #########################
		self.policy_optim.zero_grad()
		policy_loss.backward()
		self.policy_optim.step()
		self.policy_loss_history.append(policy_loss.item())

		self.Q_optim.zero_grad()
		Q_loss.backward()
		self.Q_optim.step()
		self.Q_loss_history.append(Q_loss.item())
				

	def plot_rewards(self):
		PATH = os.path.abspath(__file__)
		plt.plot(self.scores)
		plt.savefig(f"PGO/res/compatible_deterministic_PGO_score.png")
		plt.clf()

		plt.plot(self.Q_loss_history)
		plt.savefig(f"PGO/res/compatible_deterministic_PGO_critic_loss.png")
		plt.clf()

		plt.plot(self.policy_loss_history)
		plt.savefig(f"PGO/res/compatible_deterministic_PGO_actor_loss.png")
		plt.clf()


num_cores = 8
torch.set_num_interop_threads(num_cores) # Inter-op parallelism
torch.set_num_threads(num_cores) # Intra-op parallelism
env = gym.make('LunarLander-v2', render_mode=None, continuous=True)
trainer = Reinforce(env=env, n_episodes=100, gamma=0.99, buffer_max_len=100000, steps2opt=2, batch_size=64, policy_lr=1e-4, Q_lr=1e-4)
trainer.train()
trainer.plot_rewards()

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from networks.stochastic_network import Actor, Critic
from my_utils.Memory import Memory
from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from statistics import mean
import time
from agent_interface import Agent
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO_agent(Agent):
    def __init__(self, env, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, solved_reward, max_episodes, max_timesteps, update_timestep, early_stopping_window, model_path):
        self.env = env
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.solved_reward = solved_reward
        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.update_timestep = update_timestep
        self.early_stopping_window = early_stopping_window
        self.memory = Memory()
        self.n_states = state_dim
        self.n_actions = action_dim
        self.n_latent_var = n_latent_var
        self.model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), model_path))

        # actor
        self.policy = Actor(n_states=self.n_states, n_actions=self.n_actions, hidden=self.n_latent_var).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = Actor(n_states=self.n_states, n_actions=self.n_actions, hidden=self.n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # critic V function
        self.critic = Critic(input_dim=self.n_states, output_dim=1, n_hidden=self.n_latent_var)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=betas)
        self.MseLoss = nn.MSELoss()

        # log
        self.policy_loss_history = []
        self.critic_loss_history = []
        self.scores_history = []
        self.window = deque(maxlen=self.early_stopping_window)

        self.clipped_gradient_counter = [0]
        
    
    def optimize(self):   
        # Monte Carlo estimate of state rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.cat(self.memory.states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values with new policy
            new_probs = self.policy(old_states)
            new_dist = Categorical(new_probs)
            new_logprobs = new_dist.log_prob(old_actions)

            # critic
            state_values = self.critic(old_states).squeeze()
            advantages = rewards - state_values.detach()

            # Finding the ratio (pi_theta / pi_theta__old) QUESTO PASSAGGIO NASCONDE DELLA MATEMATICA !!!! => TODO: DA RIVEDERE
            ratios = torch.exp(new_logprobs - old_logprobs.detach())

            # CLIPPED SURROGATE LOSS
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages.detach()
            # MseLoss is for the update of critic, dist_entropy denotes an entropy bonus
            new_policy_loss = -torch.min(surr1, surr2)

            #TODO: REMOVE
            for i in range(len(surr2)):
                if surr1[i] != surr2[i]:
                    self.clipped_gradient_counter.append(self.clipped_gradient_counter[-1]+1)
                else: self.clipped_gradient_counter.append(self.clipped_gradient_counter[-1])

            critic_loss = 0.5*self.MseLoss(state_values, rewards)
            
            # optimize new policy
            self.policy_optimizer.zero_grad()
            new_policy_loss.mean().backward()
            self.policy_loss_history.append(new_policy_loss.mean().item())
            self.policy_optimizer.step()

            # optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def train(self):
        start_timestamp = time.time()
        timestep = 0

        # training loop
        t = trange(1, self.max_episodes+1)
        for i_episode in t:
            ep_reward = 0
            state = self.env.reset()[0]
            state = torch.tensor(state).unsqueeze(0)
            truncated = False
            terminated = False

            while not (truncated or terminated):
                timestep += 1
        
                # Running policy_old:
                with torch.no_grad():
                    probs = self.policy_old(state).squeeze()
                    dist = Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                new_state, reward, terminated, truncated, _ = self.env.step(action.numpy())

                # memory update
                self.memory.states.append(state)
                self.memory.actions.append(action)
                self.memory.logprobs.append(log_prob)
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(terminated)

                # state update
                state = torch.tensor(new_state).unsqueeze(0)
                
                # update if its time
                if timestep % self.update_timestep == 0:
                    self.optimize()
                    self.memory.clear_memory()
                    timestep = 0
                
                ep_reward += reward

                if terminated:
                    break
            
            self.scores_history.append(ep_reward)
            t.set_description(f"SCORE: {round(ep_reward, 2)}")

            # early stopping condition
            self.window.append(ep_reward)
            if int(mean(self.window)) == self.solved_reward:
                end_timestamp = time.time()
                total_time = end_timestamp - start_timestamp
                print(f"[ENVIRONMENT SOLVED in {total_time} seconds]")
                # save model
                self.save_model(self.model_path)
                t.close()
                break

        print(f"[ENVIRONMENT NOT SOLVED]")
        self.save_model(self.model_path)

    def plot(self, plot_scores=True, plot_critic_loss=True, plot_actor_loss=True):
        if plot_scores is True:
            window_size = 10
            smoothed_data = np.convolve(self.scores_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            ax = plt.gca()
            ax.set_xlim([0, 3000])
            ax.set_ylim([0, 500])
            plt.title("Score")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.savefig("PGO/res/PPO/score_0.01.png")
            plt.clf()

        if plot_critic_loss is True:
            window_size = 5
            smoothed_data = np.convolve(self.critic_loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            ax = plt.gca()
            ax.set_xlim([0, 700])
            plt.title("Value function loss")
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.savefig("critic_loss_0.01.png")
            plt.clf()

        if plot_actor_loss is True:
            window_size = 5
            smoothed_data = np.convolve(self.policy_loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            ax = plt.gca()
            ax.set_xlim([0, 700])
            plt.title("Policy function loss")
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.savefig("policy_loss_0.01.png")
            plt.clf()
        
        if plot_actor_loss is True:
            window_size = 1
            smoothed_data = np.convolve(self.clipped_gradient_counter, np.ones(window_size)/window_size, mode='valid')
            plt.plot(self.clipped_gradient_counter)
            ax = plt.gca()
            ax.set_xlim([0, 1.2e6])
            ax.set_ylim([0, 800000])
            plt.title("Clipped Gradient Counter")
            plt.xlabel("Steps")
            plt.ylabel("Clips")
            plt.savefig("clipped_gradient_0.01.png")
            plt.clf()

    def test(self, n_episodes, env):
        self.load_model(self.model_path)
        for ep in range(n_episodes):
            truncated = False
            terminated = False

            state = env.reset()[0]
            state = torch.tensor(state)

            while not (truncated or terminated):
                distr_params = self.policy(state.unsqueeze(0))

                distr = Categorical(distr_params)
			
                action = distr.sample().squeeze()

                new_state, reward, terminated, truncated, _ = env.step(action.numpy())

                state = torch.tensor(new_state)

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy = Actor(n_states=self.n_states, n_actions=self.n_actions, hidden=self.n_latent_var)
        self.policy.load_state_dict(torch.load(path))

        

def main():
    ############## Hyperparameters ##############
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    solved_reward = 500         # stop training if avg_reward > solved_reward
    max_episodes = 3000         # max training episodes
    max_timesteps = 1000        # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    early_stopping_window = 20
    model_path = model_path = "saved_models\\PPO_actor"
    #############################################

    ppo = PPO_agent(env, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, solved_reward, max_episodes, max_timesteps, update_timestep, early_stopping_window, model_path = model_path)
    ppo.train()
    ppo.plot()
    env.close()
    env = gym.make('CartPole-v1', render_mode = "human")
    ppo.test(n_episodes=5, env=env)
    env.close()

if __name__ == '__main__':
    main()  

         

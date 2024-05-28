import gymnasium as gym
from my_utils.Replay_buffer import ReplayBuffer
from networks.stochastic_network import Actor, Critic
from tqdm import trange
import torch
from matplotlib import pyplot as plt
import os
import numpy as np
from torch.distributions.categorical import Categorical
import os
from collections import deque
from statistics import mean
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class PGO_trainer():
    def __init__(self, env, gamma, episodes, actor_lr, critic_lr, buffer_max_len, solved_reward, early_stopping_window,model_path,target_model_update_rate = 1e-2, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.episodes = episodes
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.buffer_max_len = buffer_max_len
        self.batch_size = batch_size
        self.target_model_update_rate = target_model_update_rate
        self.solved_reward = solved_reward
        self.early_stopping_window = early_stopping_window
        self.model_path = model_path

        self.critic_loss_history = []
        self.actor_loss_history = []
        self.mean_history = []
        self.std_history = []
        self.scores = []
        self.window = deque(maxlen=self.early_stopping_window)

        self.n_actions = 2
        self.n_states = 4
        
        self.replay_buffer = ReplayBuffer(maxlen=buffer_max_len)

        # ACTOR
        self.actor_network = Actor(self.n_states, self.n_actions)
        self.actor_optim = torch.optim.Adam(params = self.actor_network.parameters(), lr = actor_lr)
        self.old_actor = Actor(self.n_states, self.n_actions)
        self.old_actor.load_state_dict(self.actor_network.state_dict())
        self.policy_ratios = [] 

        # CRITIC
        self.critic_network = Critic(input_dim=self.n_states+1, output_dim=1)
        self.target_critic_network = Critic(input_dim=self.n_states + 1, output_dim=1)
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())
        self.critic_optim = torch.optim.Adam(params = self.critic_network.parameters(), lr = self.critic_lr)
        self.loss_fn = torch.nn.MSELoss()

        self.ep = 0
        
    def train(self):
        start_timestamp = time.time()
        step = 0
        t = trange(self.episodes)
        for self.ep in t:
            step = 0
            truncated = False
            terminated = False

            state = self.env.reset()[0]
            state = torch.tensor(state)

            self.scores.append(0)

            while not (truncated or terminated):
                distr_params = self.actor_network(state.unsqueeze(0))

                distr = Categorical(distr_params)
			
                action = distr.sample().squeeze()	# returns a tenso

                new_state, reward, terminated, truncated, _ = self.env.step(action.numpy())	# tensor => numpy array

                self.replay_buffer.append((state, action, new_state, reward, terminated, truncated))

                self.scores[-1]+=reward
                
                state = torch.tensor(new_state)
    
            
            if len(self.replay_buffer) > self.batch_size*2:
                self.optimize()
                self.update_target_model(self.target_model_update_rate)
                self.old_actor.load_state_dict(self.actor_network.state_dict())
            t.set_description(f"Episode score: {round(self.scores[-1], 2)}, critic_loss: {round(sum(self.critic_loss_history)/(len(self.critic_loss_history)+1),2)}, actor_loss: {round(sum(self.actor_loss_history)/(len(self.actor_loss_history)+1),2)}")

            # early stopping condition
            self.window.append(self.scores[-1])
            if int(mean(self.window)) == self.solved_reward:
                end_timestamp = time.time()
                total_time = end_timestamp - start_timestamp
                print(f"[ENVIRONMENT SOLVED in {total_time} seconds]")
                # save model
                self.save_model(self.model_path)
                t.close()
                return
            
        end_timestamp = time.time()
        total_time = end_timestamp - start_timestamp
        print(f"[ENVIRONMENT NOT SOLVED. Elapsed time: {total_time} seconds]")
        self.save_model(self.model_path)

    def save_model(self, path):
        torch.save(self.actor_network.state_dict(), path)

    def load_model(self, path):
        self.actor_network = Actor(self.n_states, self.n_actions)
        self.actor_network.load_state_dict(torch.load(path))

    def test(self, n_episodes, model_path, env):
        self.load_model(model_path)
        for ep in range(n_episodes):
            truncated = False
            terminated = False

            state = env.reset()[0]
            state = torch.tensor(state)

            while not (truncated or terminated):
                distr_params = self.actor_network(state.unsqueeze(0))

                distr = Categorical(distr_params)
			
                action = distr.sample().squeeze()

                new_state, reward, terminated, truncated, _ = env.step(action.numpy())

                state = torch.tensor(new_state)
    

        
    def optimize(self):
        batch = self.replay_buffer.sample(sample_size=self.batch_size)
        
        with torch.no_grad():
            states = [x[0] for x in batch]
            actions = [x[1] for x in batch]
            next_states = [torch.tensor(x[2]) for x in batch]
            rewards = [torch.tensor(x[3], dtype=torch.float32) for x in batch]
            terminated = [torch.tensor(x[4]) for x in batch]
            
            actions_batch = torch.stack(actions).unsqueeze(1)
            states_batch = torch.stack(states)
            next_states_batch = torch.stack(next_states)
            rewards_batch = torch.stack(rewards)
            terminated_batch = torch.stack(terminated)    
            state_action_batch = torch.cat((states_batch, actions_batch), dim=1)

        ############# CRITIC OPTIMIZATION #############

        # current batch
        pred_q_batch = self.critic_network(state_action_batch).squeeze()

        # target
        with torch.no_grad():   
            distr_params = self.actor_network(states_batch)
            
            m = Categorical(distr_params)
			
            next_actions_batch = m.sample().unsqueeze(1)

            next_state_next_action_batch = torch.cat((next_states_batch, next_actions_batch), dim=1)

            next_q = self.target_critic_network(next_state_next_action_batch).squeeze()
                                
            target_q_batch = rewards_batch + ~terminated_batch * self.gamma * next_q

        critic_loss = self.loss_fn(pred_q_batch, target_q_batch)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.critic_loss_history.append(critic_loss.item())

        ############# ACTOR OPTIMIZATION #############
        old_distr_params = self.old_actor(states_batch)
        old_m = Categorical(old_distr_params)
        old_log_probs = old_m.log_prob(actions_batch)
        if self.ep > 50:
            distr_params = self.actor_network(states_batch)
            
            m = Categorical(distr_params)

            log_probs = m.log_prob(actions_batch)

            with torch.no_grad():
                self.policy_ratios.append(torch.exp(log_probs-old_log_probs).mean().item())

            # current batch
            with torch.no_grad():
                pred_q_batch = self.critic_network(state_action_batch).squeeze().detach()
                norm_q_batch = (pred_q_batch - pred_q_batch.mean()) / (pred_q_batch.std() + 1e-6)


            actor_loss = - (norm_q_batch* log_probs).sum()

            ############# BACKWARDS #############

            self.actor_optim.zero_grad()
            self.actor_loss_history.append(actor_loss.sum().item())
            actor_loss.backward()
            self.actor_optim.step()

    def plot(self, plot_scores=True, plot_critic_loss=True, plot_actor_loss=True, plot_ratios=True):
        PATH = os.path.abspath(__file__)

        if plot_scores is True:
            window_size = 10
            smoothed_data = np.convolve(self.scores, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            plt.title("Score")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.savefig("PGO/res/AC/score.png")
            plt.clf()

        if plot_critic_loss is True:
            window_size = 10
            smoothed_data = np.convolve(self.critic_loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            plt.title("Critic function loss")
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.savefig("PGO/res/AC/critic_loss.png")
            plt.clf()

        if plot_actor_loss is True:
            window_size = 10
            smoothed_data = np.convolve(self.actor_loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_data)
            plt.title("Actor function loss")
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.savefig("PGO/res/AC/actor_loss.png")
            plt.clf()

        if plot_ratios is True:
            plt.plot(self.policy_ratios)
            plt.title("New policy / Old policy")
            plt.xlabel("Step")
            plt.ylabel("ratio")
            plt.savefig("PGO/res/AC/ratios.png")
            plt.clf()
        
    def update_target_model(self, tau):
        weights = self.critic_network.state_dict()
        target_weights = self.target_critic_network.state_dict()
        for i in target_weights:
            target_weights[i] = weights[i] * tau + target_weights[i] * (1-tau)
        self.target_critic_network.load_state_dict(target_weights)
   

# PARAMETERS
env = gym.make('CartPole-v1')
gamma = 0.99
episodes = 700
actor_lr = 0.001
critic_lr= 0.001
buffer_max_len = 100000
batch_size = 256
target_model_update_rate = 0.01
solved_reward = 500
early_stopping_window = 20

model_path = "PGO/algorithms/saved_models/AC_actor"
trainer = PGO_trainer(env = env, gamma=gamma, episodes=episodes, actor_lr=actor_lr, critic_lr=critic_lr, buffer_max_len=buffer_max_len, batch_size=batch_size, target_model_update_rate = target_model_update_rate, solved_reward=solved_reward, early_stopping_window = early_stopping_window, model_path=model_path)
trainer.train()
trainer.plot(plot_scores=True, plot_actor_loss=True, plot_critic_loss=True)
env.close()
env = gym.make('CartPole-v1', render_mode = "human")
trainer.test(n_episodes=5, model_path=model_path, env=env)
env.close()
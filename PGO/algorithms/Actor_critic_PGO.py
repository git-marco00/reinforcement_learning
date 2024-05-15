import gymnasium as gym
from my_utils.Replay_buffer import ReplayBuffer
from networks.actor_network import Actor
from networks.critic_network import Critic
import random
from tqdm import trange
import torch
from matplotlib import pyplot as plt
import os
import numpy as np


class PGO_trainer():
    def __init__(self, env, gamma, episodes, actor_lr, critic_lr, buffer_max_len, steps2opt, steps2converge, mode, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.episodes = episodes
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.buffer_max_len = buffer_max_len
        self.steps2opt = steps2opt
        self.batch_size = batch_size
        self.steps2converge = steps2converge
        self.mode = mode

        self.num_opt = 0
        self.critic_loss_history = []
        self.actor_loss_history = []

        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.shape[0]
        self.scores = []
        self.replay_buffer = ReplayBuffer(maxlen=buffer_max_len)

        # ACTOR
        self.actor_network = Actor(self.n_states, self.n_actions)
        self.actor_optim = torch.optim.Adam(params = self.actor_network.parameters(), lr = actor_lr)

        # CRITIC
        self.critic_network = Critic(n_actions=self.n_actions, n_states=self.n_states)
        self.target_critic_network = Critic(n_actions=self.n_actions, n_states=self.n_states)
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())

        self.critic_optim = torch.optim.AdamW(params = self.critic_network.parameters(), lr = self.critic_lr, amsgrad = True)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
    def train(self):
        step = 0
        eps = 1
        for ep in trange(self.episodes):
            truncated = False
            terminated = False

            state = self.env.reset()[0]
            state = torch.tensor(state)
            state[5]*=2.5

            episode_score = 0

            if eps > 0.01:
                eps *= 0.992

            while not (truncated or terminated):
                #action = self.select_action(state.unsqueeze(0))

                if random.random() <= eps:
                    action = torch.tensor(self.env.action_space.sample())
                
                with torch.no_grad():
                    action = torch.argmax(self.critic_network(state.unsqueeze(0)))
            
                new_state, reward, terminated, truncated, _ = self.env.step(action.item())

                self.replay_buffer.append((state, action, new_state, reward, terminated, truncated))

                episode_score += reward
            
                step += 1
                    
                state = torch.tensor(new_state)
                state[5]*=2.5

                if step % self.steps2opt == 0 and len(self.replay_buffer) > self.batch_size*4:
                    for i in range(2):
                        self.optimize_critic()
                        #self.optimize_actor()
                        self.target_critic_network.load_state_dict(self.critic_network.state_dict())

                if step % self.steps2converge == 0:
                    pass
            
            self.scores.append(episode_score)
            
            
    def select_action(self, state):
        with torch.no_grad():
            probs = self.actor_network(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action
        
    def optimize_actor(self):
        batch = self.replay_buffer.sample(sample_size=self.batch_size)

        states = [x[0] for x in batch]
        actions = [x[1] for x in batch]
        
        states_batch = torch.stack(states)
        actions_batch = torch.cat(actions)

        # non posso usare le azioni che ho fatto nel passato, perchè nel mentre la mia rete actor
        # si è aggiornata ed è migliorata

        probs = self.actor_network(states_batch)
        dist = torch.distributions.Categorical(probs)
        log_probs_batch = dist.log_prob(actions_batch)

        with torch.no_grad():
            q_values_all_actions = self.critic_network(states_batch)
            q_values = torch.gather(q_values_all_actions, 1, actions_batch.unsqueeze(dim=1))
        
        loss = - log_probs_batch * q_values

        self.actor_optim.zero_grad()
        loss.mean().backward()
        self.actor_loss_history.append(loss.mean().item())
        self.actor_optim.step()   
  
    def optimize_critic(self):
        self.num_opt +=1
        batch = self.replay_buffer.sample(sample_size=self.batch_size)

        states = [x[0] for x in batch]
        actions = [x[1] for x in batch]
        next_states = [torch.tensor(x[2]) for x in batch]
        rewards = [torch.tensor(x[3], dtype=torch.float32) for x in batch]
        terminated = [torch.tensor(x[4]) for x in batch]
        
        actions_batch = torch.stack(actions).unsqueeze(dim=1)
        states_batch = torch.stack(states)
        next_states_batch = torch.stack(next_states)
        rewards_batch = torch.stack(rewards)
        terminated_batch = torch.stack(terminated)


        # Q(s,a,w) = reward + gamma * max a' Q(s', a', w-) - Q(s, a, w)
        # questo perchè io voglio che la mia Q (ovvero l'actor) abbia un valore che si avvicini il più possibile
        # alla reward che ho ottenuto effettivamente facendo quell'azione + tutto ciò che ottengo dopo
        # ma utilizzando una rete stabile ovver Q(w-)

        # current batch
        q_batch = self.critic_network(states_batch)
        current_batch = torch.gather(q_batch, 1, actions_batch).squeeze() # Q(s, a, w)

        # target
        if self.mode == "simple_DQN":
            with torch.no_grad():
                next_q_batch = self.target_critic_network(next_states_batch)
                next_max_q_batch,_ = torch.max(next_q_batch, dim=1)
                target_batch = rewards_batch + ~terminated_batch * self.gamma * next_max_q_batch    # reward + gamma * max a' Q(s', a', w-)
        
        elif self.mode == "double_DQN":
            with torch.no_grad():
                next_q_batch_theta_meno = self.critic_network(next_states_batch)
                best_actions_batch = torch.argmax(next_q_batch_theta_meno, dim=1).unsqueeze(1)
                next_q_batch_theta = self.target_critic_network(next_states_batch)
                next_q_batch = torch.gather(next_q_batch_theta, 1, best_actions_batch).squeeze()
                target_batch = rewards_batch + ~terminated_batch * self.gamma * next_q_batch

        loss = self.loss_fn(current_batch, target_batch)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        self.critic_loss_history.append(loss.item())


        
    def plot_rewards(self):
        PATH = os.path.abspath(__file__)
        plt.plot(self.scores)
        plt.savefig("PGO/res/PGO_AC_DDQN_score_mine.png")
        plt.clf()

        window_size = 10
        smoothed_data = np.convolve(self.critic_loss_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_data)
        plt.savefig("PGO/res/PGO_AC_DDQN_critic_loss_mine.png")
        plt.clf()

        """window_size = 10
        smoothed_data = np.convolve(self.actor_loss_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_data)
        plt.savefig("PGO/res/PGO_AC_DDQN_actor_loss_mine.png")
        plt.clf()"""
   
    
env = gym.make('LunarLander-v2')
trainer = PGO_trainer(env = env, gamma=0.99, episodes=50, actor_lr=3e-3, critic_lr=0.01, buffer_max_len=100000, steps2opt=4, steps2converge=8, mode="double_DQN", batch_size=32)

trainer.train()
trainer.plot_rewards()
env.close()
print(trainer.num_opt)


    

    

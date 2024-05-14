import gymnasium as gym
from my_utils.Replay_buffer import ReplayBuffer
from networks.actor_network import Actor
from networks.linear_critic import LinearCritic
import random
from tqdm import trange
import torch
from matplotlib import pyplot as plt
import os


class PGO_trainer():
    def __init__(self, env, gamma, episodes, actor_lr, critic_lr, buffer_max_len, steps2opt, batch_size=32):
        self.env = env
        self.gamma = gamma
        self.episodes = episodes
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.buffer_max_len = buffer_max_len
        self.steps2opt = steps2opt
        self.batch_size = batch_size

        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.shape[0]
        self.scores = []
        self.replay_buffer = ReplayBuffer(maxlen=buffer_max_len)

        # ACTOR
        self.actor_network = Actor(self.n_states, self.n_actions)
        self.actor_optim = torch.optim.Adam(params = self.actor_network.parameters(), lr = actor_lr)

        # CRITIC
        self.linear_critic_network = LinearCritic(n_actions=self.n_actions, n_states=self.n_states)
        self.critic_optim = torch.optim.Adam(params = self.linear_critic_network.parameters(), lr = self.critic_lr)
        
        self.loss_fn = torch.nn.L1Loss()

        
    def train(self):
        step = 0
        for ep in trange(self.episodes):
            self.scores.append(0)
            transitions = []

            truncated = False
            terminated = False

            state = self.env.reset()[0]
            state = torch.Tensor(state)

            while not (truncated or terminated):
                action, log_prob = self.select_action(state.unsqueeze(0))
            
                new_state, reward, terminated, truncated, _ = self.env.step(action.item())

                transitions.append((state, action, log_prob))

                self.replay_buffer.append((state, action, log_prob, new_state, reward, terminated, truncated))

                self.scores[ep]+=reward

                step += 1
                if step % self.steps2opt == 0 and len(self.replay_buffer) > self.batch_size:
                    self.optimize_critic()

                state = torch.Tensor(new_state)
            
            self.optimize_actor(transitions)
            transitions.clear()

    def select_action(self, state):
        probs = self.actor_network(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def optimize_actor(self, transitions):

        states = [x[0] for x in transitions]
        actions = [x[1] for x in transitions]
        log_probs = [x[2] for x in transitions]

        states_batch = torch.stack(states)
        log_probs_batch = torch.cat(log_probs)
        actions_batch = torch.stack(actions)

        q_values_all_actions = self.linear_critic_network(states_batch)

        q_values = torch.gather(q_values_all_actions, 1, actions_batch)
        
        loss = - log_probs_batch * q_values.detach()
        
        self.actor_optim.zero_grad()
        loss.mean().backward()
        self.actor_optim.step()   
  

    def optimize_critic(self):
        batch = self.replay_buffer.sample(sample_size=self.batch_size)

        states = [x[0] for x in batch]
        next_states = [torch.tensor(x[3]) for x in batch]
        rewards = [torch.tensor(x[4]) for x in batch]
        terminated = [torch.tensor(x[5]) for x in batch]
        
        states_batch = torch.stack(states)
        next_states_batch = torch.stack(next_states)
        rewards_batch = torch.stack(rewards)
        terminated_batch = torch.stack(terminated)

        # Q(s,a,w) = reward + gamma * Q(s+1, a+1, w)

        # q values
        q_batch = self.linear_critic_network(states_batch)
        q_values, _ = torch.max(q_batch, dim=1)     # MA SIAMO SICURI CHE QUA SIA COSì????? Non dovrei forse selezionare 
                                                    # in base dall'azione che ho preso?

        # target
        next_q_batch = self.linear_critic_network(next_states_batch)
        next_max_q_batch,_ = torch.max(next_q_batch, dim=1)

        target = rewards_batch + ~terminated_batch * self.gamma * next_max_q_batch
        
        loss = self.loss_fn(q_values, target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        
    def plot_rewards(self):
        PATH = os.path.abspath(__file__)
        x = range(0, self.episodes)
        plt.plot(x, self.scores)
        plt.savefig(f"PGO/res/actor_critic_{self.env.unwrapped.spec.id}.png")
    
    def test(self, env, n_episodes=5):
        for i in range (n_episodes):
            state = env.reset()[0]
            truncated = False
            terminated = False
            while not (truncated or terminated):
                action = self.policy_network(self.state_OHE(state)).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)
     
env = gym.make('LunarLander-v2')
trainer = PGO_trainer(env = env, gamma=0.99, episodes=500, actor_lr=1e-3, critic_lr=1e-3, buffer_max_len=10000, steps2opt=10)
trainer.train()
trainer.plot_rewards()
env.close()


    

    

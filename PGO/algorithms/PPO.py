import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from networks.stochastic_network import Actor, Critic
from my_utils.Memory import Memory
from tqdm import trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO():
    def __init__(self, env, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, solved_reward, log_interval, max_episodes, max_timesteps, update_timestep):
        self.env = env
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.solved_reward = solved_reward
        self.log_interval = log_interval
        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.update_timestep = update_timestep
        self.memory = Memory()

        
        self.policy = Actor(n_states=state_dim, n_actions=action_dim, hidden=n_latent_var).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = Actor(n_states=state_dim, n_actions=action_dim, hidden=n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.critic = Critic(input_dim=state_dim, output_dim=1, n_hidden=n_latent_var)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=betas)

        self.MseLoss = nn.MSELoss()
    
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

            critic_loss = 0.5*self.MseLoss(state_values, rewards)
            
            # optimize new policy
            self.policy_optimizer.zero_grad()
            new_policy_loss.mean().backward()
            self.policy_optimizer.step()

            # optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.mean().backward()
            self.critic_optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def train(self):
        timestep = 0

        # training loop
        t = trange(1, self.max_episodes+1)
        for i_episode in t:
            ep_reward = 0
            state = self.env.reset()[0]
            truncated = False
            terminated = False

            while not (truncated or terminated):
                state = torch.tensor(state).unsqueeze(0)
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

                state = torch.tensor(new_state)
                
                # update if its time
                if timestep % self.update_timestep == 0:
                    self.optimize()
                    self.memory.clear_memory()
                    timestep = 0
                
                ep_reward += reward

                if terminated:
                    break
                    
            t.set_description(f"SCORE: {round(ep_reward, 2)}")

        
def main():
    ############## Hyperparameters ##############
    env_name = "CartPole-v1"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    solved_reward = 500         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 1000        # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################
    
    ppo = PPO(env, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, solved_reward, log_interval, max_episodes, max_timesteps, update_timestep)
    ppo.train()

if __name__ == '__main__':
    main()     
    
         

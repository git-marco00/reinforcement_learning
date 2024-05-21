# Imports
from typing import Iterable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy
import numpy as np
import pandas as pd
import random
from PIL import Image
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Callable
from collections import namedtuple
import itertools
from tqdm import trange

class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        """
        Initialize the Critic network.
        
        :param obs_dim: dimention of the observations
        :param num_actions: dimention of the actions
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()


    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out
    
class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_low: np.array, action_high: np.array):
        """
        Initialize the Actor network.
        
        :param obs_dim: dimention of the observations
        :param num_actions: dimention of the actions
        """
        super(Actor, self).__init__()
        # We are registering scale and bias as buffers so they can be saved and loaded as part of the model.
        # Buffers won't be passed to the optimizer for training!
        self.register_buffer(
            "action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32)
        )
        
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)

    def forward(self, obs: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Forward pass of the actor network.
        
        return: mean_action, log_prob_action
        """
        x = F.relu(self.fc1(obs))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) 
        dist = torch.distributions.Normal(mu, std)

        action = dist.rsample()
        log_prob = dist.log_prob(action)

        # Enforcing action bounds
        adjusted_action = torch.tanh(action) * self.action_scale + self.action_bias
        adjusted_log_prob = log_prob - torch.log(self.action_scale * (1-torch.tanh(action).pow(2)) + 1e-6)
        return adjusted_action, adjusted_log_prob
        

        
class ReplayBuffer:
    def __init__(self, max_size: int):
        """
        Create the replay buffer.

        :param max_size: Maximum number of transitions in the buffer.
        """
        self.data = []
        self.max_size = max_size
        self.position = 0

    def __len__(self) -> int:
        """Returns how many transitions are currently in the buffer."""
        return len(self.data)

    def store(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, terminated: torch.Tensor):
        """
        Adds a new transition to the buffer. When the buffer is full, overwrite the oldest transition.

        :param obs: The current observation.
        :param action: The action.
        :param reward: The reward.
        :param next_obs: The next observation.
        :param terminated: Whether the episode terminated.
        """
        if len(self.data) < self.max_size:
            self.data.append((obs, action, reward, next_obs, terminated))
        else:
            self.data[self.position] = (obs, action, reward, next_obs, terminated)   
        self.position = (self.position + 1) % self.max_size         

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample a batch of transitions uniformly and with replacement. The respective elements e.g. states, actions, rewards etc. are stacked

        :param batch_size: The batch size.
        :returns: A tuple of tensors (obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch), where each tensors is stacked.
        """
        return [torch.stack(b) for b in zip(*random.choices(self.data, k=batch_size))]

def update_critics(
        q1: nn.Module,
        q1_target: nn.Module,
        q1_optimizer: optim.Optimizer,
        q2: nn.Module,
        q2_target: nn.Module,
        q2_optimizer: optim.Optimizer,
        actor_target: nn.Module,
        log_ent_coef: torch.Tensor,
        gamma: float,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        tm: torch.Tensor,
        critic_loss_history: list
    ):
    """
    Update both of SAC's critics for one optimizer step.

    :param q1: The first critic network.
    :param q1_target: The target first critic network.
    :param q1_optimizer: The first critic's optimizer.
    :param q2: The second critic network.
    :param q2_target: The target second critic network.
    :param q2_optimizer: The second critic's optimizer.
    :param actor: The actor network.
    :param actor_target: The target actor network.
    :param actor_optimizer: The actor's optimizer.
    :param gamma: The discount factor.
    :param obs: Batch of current observations.
    :param act: Batch of actions.
    :param rew: Batch of rewards.
    :param next_obs: Batch of next observations.
    :param tm: Batch of termination flags.

    """
    # Calculate the target
    with torch.no_grad():
        next_action, next_action_log_prob = actor_target(torch.as_tensor(next_obs))
        next_q1_target = q1_target(torch.as_tensor(next_obs), next_action)
        next_q2_target = q2_target(torch.as_tensor(next_obs), next_action)
        next_min_q_target = torch.min(next_q1_target, next_q2_target)

        tm = tm.view(-1, 1).float()  # Ensure tm has the right shape
        target = rew.unsqueeze(1) + gamma * (1.0 - tm) * next_min_q_target - log_ent_coef.exp() * next_action_log_prob 

    
    for q, q_target, optimizer in [(q1, q1_target, q1_optimizer), (q2, q2_target, q2_optimizer)]:
        q_val = q(obs, act).repeat(1,2)
        loss = F.mse_loss(q_val, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    critic_loss_history.append(loss.item())


def update_actor(
    q1: nn.Module,
    q2: nn.Module,
    actor: nn.Module,
    actor_optimizer: optim.Optimizer,
    obs: torch.Tensor,
    log_ent_coef: torch.Tensor, 
    actor_loss_history: list
    ):
    """
    Update the SAC's Actor network for one optimizer step.

    :param critic: The critic network.
    :param actor: The actor network.
    :param actor_optimizer: The actor's optimizer.
    :param obs: Batch of current observations.

    """
    # Actor Update
    action, action_log_prob = actor(obs)
    entropy = - log_ent_coef.exp() * action_log_prob
    q1, q2 = q1(obs, action), q2(obs, action)
    q1_q2 = torch.cat([q1, q2], dim=1)
    min_q = torch.min(q1_q2, 1, keepdim=True)[0]
    actor_loss = (- min_q - entropy).mean()
    actor_loss_history.append(actor_loss.item())
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

def update_entropy_coefficient(
        actor: nn.Module,
        log_ent_coef: torch.Tensor,
        target_entropy: float,
        ent_coef_optimizer: optim.Optimizer,
        obs: torch.Tensor,
    ):
    """
    Automatic update for entropy coefficient (alpha)

    :param actor: the actor network.
    :param log_ent_coef: tensor representing the log of entropy coefficient (log_alpha).
    :param target_entropy: tensor representing the desired target entropy.
    :param ent_coef_optimizer: torch optimizer for entropy coefficient.
    :param obs: current batch observation.
    """
    _, action_log_prob = actor(obs)
    ent_coef_loss = -(log_ent_coef.exp() * (action_log_prob + target_entropy).detach()).mean()
    ent_coef_optimizer.zero_grad()
    ent_coef_loss.backward()
    ent_coef_optimizer.step()

def polyak_update(
    params: Iterable[torch.Tensor],
    target_params: Iterable[torch.Tensor],
    tau: float,
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:

    :param params: parameters of the original network (model.parameters())
    :param target_params: parameters of the target network (model_target.parameters())
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) 1 -> Hard update, 0 -> No update
    """
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


class SACAgent:
    def __init__(self,
            env,
            gamma=0.99,
            lr=0.001, 
            batch_size=64,
            tau=0.005,
            maxlen=100_000,
            target_entropy=-1.0,
        ):
        """
        Initialize the SAC agent.

        :param env: The environment.
        :param exploration_noise.
        :param gamma: The discount factor.
        :param lr: The learning rate.
        :param batch_size: Mini batch size.
        :param tau: Polyak update coefficient.
        :param max_size: Maximum number of transitions in the buffer.
        """

        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.target_entropy=target_entropy

        # Initialize the Replay Buffer
        self.buffer = ReplayBuffer(maxlen)

        # Initialize two critic and one actor network
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.q1 = Critic(obs_dim, action_dim)
        self.q2 = Critic(obs_dim, action_dim)
        self.actor = Actor(obs_dim, action_dim, env.action_space.low, env.action_space.high)
        self.log_ent_coef = torch.zeros(1, requires_grad=True)

        # Initialze two target critic and one target actor networks and load the corresponding state_dicts
        self.q1_target = Critic(obs_dim, action_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target = Critic(obs_dim, action_dim)
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor_target = Actor(obs_dim, action_dim, env.action_space.low, env.action_space.high)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Create ADAM optimizer for the Critic and Actor networks
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.ent_coef_optimizer = optim.Adam([self.log_ent_coef], lr=lr)

        self.critic_loss_history = []
        self.actor_loss_history = []


    def train(self, num_episodes: int) -> EpisodeStats:
        """
        Train the SAC agent.

        :param num_episodes: Number of episodes to train.
        :returns: The episode statistics.
        """
        # Keeps track of useful statistics
        stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
        )
        current_timestep = 0

        t = trange(num_episodes)
        for i_episode in t:
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0:
                print(f'Episode {i_episode + 1} of {num_episodes}  Time Step: {current_timestep}')

            # Reset the environment and get initial observation
            obs, _ = self.env.reset()
            
            for episode_time in itertools.count():
                # Choose action and execute
                with torch.no_grad():
                    action, _ = self.actor(torch.as_tensor(obs).float())
                    action = action.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] += 1

                # Store sample in the replay buffer
                self.buffer.store(
                    torch.as_tensor(obs, dtype=torch.float32),
                    torch.as_tensor(action),
                    torch.as_tensor(reward, dtype=torch.float32),
                    torch.as_tensor(next_obs, dtype=torch.float32),
                    torch.as_tensor(terminated),
                )

                # Sample a mini batch from the replay buffer
                obs_batch, act_batch, rew_batch, next_obs_batch, tm_batch = self.buffer.sample(self.batch_size)
                
                # Update the Critic network
                update_critics(self.q1, self.q1_target, self.q1_optimizer, self.q2, self.q2_target, self.q2_optimizer, self.actor_target, 
                               self.log_ent_coef, self.gamma, obs_batch, act_batch, rew_batch, next_obs_batch, tm_batch, self.critic_loss_history)

                # Update the Actor network
                update_actor(
                    self.q1,
                    self.q2,
                    self.actor,
                    self.actor_optimizer,
                    obs_batch.float(),
                    self.log_ent_coef,
                    self.actor_loss_history
                )
                
                # Update Entropy Coefficient
                update_entropy_coefficient(
                    self.actor,
                    self.log_ent_coef,
                    self.target_entropy,
                    self.ent_coef_optimizer,
                    obs_batch.float(),
                )

                # Update the target networks via Polyak Update
                polyak_update(self.q1.parameters(), self.q1_target.parameters(), self.tau)
                polyak_update(self.q2.parameters(), self.q2_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                
                current_timestep += 1

                # Check whether the episode is finished
                if terminated or truncated or episode_time >= 500:
                    break
                obs = next_obs
            t.set_description(f"Episode score: {round(stats.episode_rewards[i_episode], 2)}, critic_loss: {round(sum(self.critic_loss_history)/(len(self.critic_loss_history)+1),2)}, actor_loss: {round(sum(self.actor_loss_history)/(len(self.actor_loss_history)+1),2)}")
        return stats



# Choose your environment
env = gym.make("LunarLander-v2",
    continuous = True,
    gravity = -10.0,
    render_mode="rgb_array")
# Print observation and action space infos
print(f"Training on {env.spec.id}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}\n")

# Hyperparameters, Hint: Change as you see fit
LR = 0.001
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100_000
TAU = 0.005
NUM_EPISODES = 250
DISCOUNT_FACTOR = 0.99
TARGET_ENTROPY = -1.0

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Train SAC
agent = SACAgent(
    env, 
    gamma=DISCOUNT_FACTOR,
    lr=LR,
    batch_size=BATCH_SIZE,
    tau=TAU,
    maxlen=REPLAY_BUFFER_SIZE,
    target_entropy=TARGET_ENTROPY,
)
stats = agent.train(NUM_EPISODES)



"""# save the trained actor
torch.save(agent.actor, "sac_actor.pt")

# loading the trained actor
loaded_actor = torch.load("sac_actor.pt")
loaded_actor.eval()
print(loaded_actor)

"""
smoothing_window=3
fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

# Plot the episode length over time
ax = axes[0]
ax.plot(stats.episode_lengths)
ax.set_xlabel("Episode")
ax.set_ylabel("Episode Length")
ax.set_title("Episode Length over Time") 

# Plot the episode reward over time
ax = axes[1]
rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
ax.plot(rewards_smoothed)
ax.set_xlabel("Episode")
ax.set_ylabel("Episode Reward (Smoothed)")
ax.set_title(f"Episode Reward over Time\n(Smoothed over window size {smoothing_window})")

fig.savefig("SAC.png")



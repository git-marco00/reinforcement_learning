from agents.PPO import PPO_agent
import gymnasium as gym

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

ppo = PPO_agent(env, state_dim, action_dim, n_latent_var, lr,
                 betas, gamma, K_epochs, eps_clip, solved_reward,
                   max_episodes, max_timesteps, update_timestep,
                     early_stopping_window, model_path = model_path)

ppo.train()
ppo.plot()
env.close()
env = gym.make('CartPole-v1', render_mode = "human")
ppo.test(n_episodes=5, env=env)
env.close()
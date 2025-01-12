import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import time

import gymnasium as gym
import gymnasium_robotics

import numpy as np
import json
import pickle

from HGR import ReplayBuffer
from utils import *

# Define the actor-critic model
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim = 256):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim + goal_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

        # initialization
        for layer in [self.linear1, self.linear2, self.linear3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state, goal):
        #  Metti qualche batch normalization
        x = torch.cat([state, goal], dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))                 # tanh final activation
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim = 256):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim + goal_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # initialization
        for layer in [self.linear1, self.linear2, self.linear3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)             
        return x

class Policy(nn.Module):
    """implement DDPG with HGR buffer
    
    train on GPU:
    -ensure that input state is on the device: state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    -ensure the ActorCritic network is on teh device: model = ActorCritic(state_dim, action_dim).to(device)
    -Perform your training loop while ensuring that both your model and data are on the GPU during forward and backward passes.
    
    use: -target_actor
         -target_critic
    """

    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Policy, self).__init__()
        self.device = device

        # initialize the environment
        self.env = gym.make("FetchReach-v3")
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.state_dim = self.obs_space['observation'].shape[0]
        self.action_dim = self.action_space.shape[0] - 1 
        self.goal_dim = self.obs_space['desired_goal'].shape[0]

        # initialize the Actor and the Critic 
        self.actor = Actor(state_dim=self.state_dim, action_dim=self.action_dim, goal_dim=self.goal_dim ).to(self.device)
        self.target_actor = Actor(state_dim=self.state_dim, action_dim=self.action_dim, goal_dim=self.goal_dim ).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim=self.state_dim, action_dim=self.action_dim, goal_dim=self.goal_dim ).to(self.device)
        self.target_critic = Critic(state_dim=self.state_dim, action_dim=self.action_dim, goal_dim=self.goal_dim ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # HYPERPARAMETERS
        self.epochs = 20000                         # n. of epochs during training. It is egual to the n. of episodes 
        self.batch_size = 100                         # batch size
        self.update_freq = 100                        # defines the network update frequency during training
        self.H = 50                                 # the horizon of one episode. Keep in mind the episode ends if we collect negative rewards for 50 consecutive steps
        self.rho = 0.05                             # defines the distance to the goal in which the reward is positive. Used in compute_reward
        self.gamma = 0.98                           # discount factor
        self.save_freq = 500                          # determines the after how much episodes the model is saved
        self.tau = 0.05                            # parameter for the soft update

        # initialize the buffer
        self.replay_buffer = ReplayBuffer(capacity=1000000, episode_horizon=self.H)#10000)

        # Initialize the logs for the evaluation
        self.training_logs = {
                "n_episodes": 0,
                "reward_per_episode": [],
                "actor_loss": [],
                "critic_loss": [],
                "training_time": []
            }
        self.start_epoch = 1

    def act(self, observation):
        """ used only in the main.py it gets one state as input and return the actions"""
        self.target_actor.eval()
        
        # preprocessing
        state =  torch.tensor(observation['observation'], dtype=torch.float32).unsqueeze(0)
        desired_goal =  torch.tensor(observation['desired_goal'], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            actions = np.array(self.target_actor(state, desired_goal))

        actions = np.hstack([actions, np.zeros((actions.shape[0], 1))])                # add 0 as the 4-th actions (in Reach task it is useless)
        #print('actions: ', actions[0])
        return actions[0]

    def noisy_action(self, observation, noise_factor, noise_type ='Uniform'):               # noise = {Uniform, Gaussian, Ornstein-Uhlenbeck}
        self.actor.eval()

        # preprocessing
        state =  torch.tensor(observation['observation'], dtype=torch.float32).unsqueeze(0).to(self.device)
        desired_goal =  torch.tensor(observation['desired_goal'], dtype=torch.float32).unsqueeze(0).to(self.device)
    
        with torch.no_grad():
            actions = np.array(self.actor(state, desired_goal).cpu())[0]

        # Improvement use Ornstein-Uhlenbeck noise
        if noise_type == 'Uniform':
            # Uniform noise in the range [-1, 1]
            noise = np.random.uniform(-1, 1, size=actions.shape) * noise_factor
        if noise_type == 'Gaussian':
            # Gaussian noise with mean 0 and standard deviation 0.2
            noise = np.random.normal(0, 0.2, size=actions.shape) * noise_factor         
        if noise_type == 'Ornstein-Uhlenbeck':
            pass
            # Ornstein-Uhlenbeck process
            # Implement ...
            # ou_state += theta * (-ou_state) * self.dt + \
            #                  self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
            # noise = self.ou_state * self.noise_factor      

        # Implement epsilon greedy...

        actions += noise
        return actions
    
    def rollout(self, noise_factor=1, decay=0.999):
        obs, _ = self.env.reset()                                                      # obs contains the state and the goal
        done = False

        total_reward = 0
        for i in range(self.H):
            # Sample an action
            action = self.noisy_action(obs, noise_factor)
            action_4dim = np.hstack([action, np.zeros(1)])                             # append the last element as 0 to get a 4-dim action
            next_obs, reward, terminated, truncated, info = self.env.step(action_4dim)
            done = False if i != self.H -1 else True

            noise_factor *= decay

            # Store transition
            self.replay_buffer.push(obs, action, next_obs, done)

            obs = next_obs
            total_reward += reward
        return total_reward                                                             # for the evaluation metric

    def compute_reward(self, next_obs, new_goal):
        ee_position = next_obs[:-7]
        distance_form_the_goal = np.linalg.norm(np.array(ee_position) - np.array(new_goal))
        # print('distance_form_the_goal',distance_form_the_goal)
        if distance_form_the_goal < self.rho:
            return 0
        else:
            return -1

    def train(self, lr_actor = 1e-3, lr_critic =1e-3, l2_lambda=0):     # ATTENTION: l2_lambda=1 but it dominates the loss function
        # Load the weights if model.pt exists
        if os.path.exists('model.pt'):
            self.load(load_for_training=True)                                                                 # load target actor and critic
            self.actor.load_state_dict(self.target_actor.state_dict())                  # copy the weights in the actor network
            self.critic.load_state_dict(self.target_critic.state_dict())                # copy the weights in the critic network
            self.target_actor.to(self.device)
            self.target_critic.to(self.device)
            self.actor.to(self.device)
            self.critic.to(self.device)

        # Initialize the optimizers for actor and critic
        actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        start_time=time.time()
        # Training loop
        for epoch in range(self.start_epoch, self.epochs + 1):                # for each episode
            
            # Rollout phase
            total_reward = self.rollout()

            # Batch computation with update frequency
            if epoch % self.update_freq == 0:
                self.replay_buffer.set_probabilities()                        # set the probabilities such that they sum up to 1 before sampling 

                # set the actor-critic in training mode 
                self.actor.train()
                self.critic.train()

                critic_loss = []
                actor_loss = []
                importance_sampling = []
                for k in range (self.batch_size):
                    episode_idx = self.replay_buffer.sample_episode()                                             # sample an episode
                    experience, new_goal, j, i = self.replay_buffer.sample_experience_and_goal(episode_idx)       # sample an experience with an hindsight goal according to the Future startegy
                                                                                                                  # here experience = (sj,aj,sj+1,delta_array, done)
                    if experience[4] == True: raise ValueError('done=True?????')

                    obs = torch.tensor(experience[0]['observation'], dtype=torch.float32).unsqueeze(0).to(self.device)
                    action = torch.tensor(experience[1],dtype=torch.float32).unsqueeze(0).to(self.device)  
                    next_obs = torch.tensor(experience[2]['observation'],dtype=torch.float32).unsqueeze(0).to(self.device)  
                    new_goal = torch.from_numpy(new_goal).to(dtype=torch.float32).unsqueeze(0).to(self.device)  

                    new_reward = self.compute_reward(next_obs.squeeze(0).cpu(),new_goal.squeeze(0).cpu()) 

                    # Compute importance sampling
                    w = self.replay_buffer.get_importance_sampling(episode_idx, j, i)
                    importance_sampling.append(w)

                    with torch.no_grad():
                        target_mu = self.target_actor(next_obs, new_goal).detach()                                       # mu(s_(j+1) ||g_i)
                        next_Q_value = self.target_critic(next_obs, target_mu, new_goal).detach()                        # Q_target(s_(j+1), mu(s_(j+1) ||g_i) || g_i)
                    
                    Q_value = self.critic(obs, action, new_goal)                                                         # Q(s_j, a_j || g_i)
                    
                    # Compute TD error
                    delta = (new_reward + self.gamma * next_Q_value - Q_value).squeeze(0)
                    
                    # Insert delta in buffer and Update the probabilities
                    self.replay_buffer.update_delta_and_probs(delta.detach().item(), episode_idx, j, i)

                    # Compute losses
                    critic_loss_k = w * delta ** 2

                    mu = self.actor(obs, new_goal)
                    # L2-regularization
                    l2_reg = l2_lambda * sum(torch.sum(param ** 2) for param in self.actor.parameters())
                    actor_loss_k = - self.critic(obs, mu, new_goal) + l2_reg

                    critic_loss.append(critic_loss_k)
                    actor_loss.append(actor_loss_k)

                # Update networks
                max_w = max(importance_sampling)
                actor_loss = [loss / max_w for loss in actor_loss]
                critic_loss = [loss / max_w for loss in critic_loss]

                actor_loss = torch.mean(torch.stack(actor_loss))
                critic_loss = torch.mean(torch.stack(critic_loss))

                # ## (Optional) clamp the actor gradients for robustness
                # for param in self.actor.parameters():
                #     param.grad = torch.clamp(param.grad, -10, 10)

                # Actor update
                actor_optimizer.zero_grad()
                actor_loss.backward()  
                actor_optimizer.step()

                # Critic update
                critic_optimizer.zero_grad()
                critic_loss.backward()  
                critic_optimizer.step()

                # Update target networks
                self.soft_update(self.target_actor, self.actor)
                self.soft_update(self.target_critic, self.critic)

                self.training_logs['actor_loss'].append(actor_loss.detach().item())
                self.training_logs['critic_loss'].append(critic_loss.detach().item())

                print(f"Epoch {epoch}/{self.epochs}, Actor loss: {actor_loss.item()}, , Critic loss: {critic_loss.item()}")
            
            self.training_logs['reward_per_episode'].append(total_reward)

            # save after tot epochs
            if (epoch) % self.save_freq == 0:
                current_time = time.time()
                self.training_logs['training_time'].append((current_time-start_time)/3600)
                self.training_logs['n_episodes'] = epoch
                self.save()
                
                self.training_logs = {
                "n_episodes": 0,
                "reward_per_episode": [],
                "actor_loss": [],
                "critic_loss": [],
                "training_time":[]
            }

    def soft_update(self, target_net, main_net):
        # Soft update for target net
        for target_param, local_param in zip(target_net.parameters(), main_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self):
        # Save the model
        torch.save({
            'actor': self.target_actor.state_dict(),
            'critic': self.target_critic.state_dict()
                    }, 'model.pt')
        
        # Save the logs
        file_path = 'training_logs.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as fl:
                old_data = json.load(fl)
            old_data['training_time'].append(old_data['training_time'][-1]+self.training_logs['training_time'][-1])
            old_data['n_episodes'] = self.training_logs['n_episodes']
            old_data['reward_per_episode'].extend(self.training_logs['reward_per_episode'])
            old_data['actor_loss'].extend(self.training_logs['actor_loss'])
            old_data['critic_loss'].extend(self.training_logs['critic_loss'])
        else:
            old_data = self.training_logs

        with open(file_path, 'w') as f:
            json.dump(old_data, f, indent=4)

        # Save the buffer
        with open('buffer.pkl', 'wb') as file:
            pickle.dump(self.replay_buffer, file)


    def load(self, load_for_training = False):
        # Load the model
        checkpoint = torch.load('model.pt', map_location=self.device)
        self.target_actor.load_state_dict(checkpoint['actor'])
        self.target_critic.load_state_dict(checkpoint['critic'])
        print("weights loaded for actor and critic!")

        # For training:
        if load_for_training:
            # Load the current epoch
            file_path = 'training_logs.json'
            if os.path.exists(file_path):
                with open(file_path, 'r') as f_load:
                    logs = json.load(f_load)
                    self.start_epoch = logs['n_episodes'] + 1

            # Load the buffer
            if os.path.exists('buffer.pkl'):
                with open('buffer.pkl', 'rb') as file_load:
                    self.replay_buffer = pickle.load(file_load)
                    print('buffer loaded')
                    print('len buffer', len(self.replay_buffer.buffer))
                    print('max_delta', self.replay_buffer.max_delta)

    def plot_training_logs(self):
        try:
            with open('training_logs.json', 'r') as f:
                logs = json.load(f)
            
            success_rate = []
            success = 0
            for i,rew in enumerate(logs['reward_per_episode']):
                if rew > -50:
                    success +=1
                if (i+1) % self.save_freq == 0:
                    success_rate.append(success/logs['n_episodes'])

            plot_logs(logs['reward_per_episode'], 
                      success_rate, 
                      logs['training_time'],
                      logs['actor_loss'],
                      logs['critic_loss'],
                      self.update_freq
                      )

        except FileNotFoundError:
            print("Error: File 'training_logs.json' not found.")

    def test(self, n_episodes=100):
        agent.load()

        env = gym.make("FetchReach-v3", max_episode_steps=50)
            
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            while not done:
                action = agent.act(s)
                
                s, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
            
            rewards.append(total_reward)
            
        print(f'Mean Reward in {n_episodes} episodes:', np.mean(rewards))
        plot_test(rewards)

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    


if __name__ == "__main__":
    agent = Policy()
    
    # FOR DEBUGGING
    #agent.train()

    # plot logs
    agent.plot_training_logs()

    # test results
    #agent.test()

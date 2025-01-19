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
from normalizer import Normalizer

# Define the actor-critic model
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim=320, hidden_dim2 = 256):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim + goal_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, action_dim)

        # initialization
        for layer in [self.linear1, self.linear2]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.zeros_(self.linear3.bias)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        x = nn.LeakyReLU()(self.linear1(x))
        x = nn.LeakyReLU()(self.linear2(x))
        x = nn.Tanh()(self.linear3(x))                 # tanh final activation
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
        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))
        x = self.linear3(x)            
        return x

class Policy(nn.Module):
    """This class implements the DDPG learning algorithm with HGR strategy.
    
    main methods:
    - train: train the agent using the DDPG + HGR training algorithm 
    - save: save the weights of Actor/Critic, the Replay Buffer and the training logs
    - load: load the weights of Actor/Critic, the Replay Buffer and the training logs
    - plot_training_logs: plot the training logs.
    """

    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Policy, self).__init__()
        self.device = device

        # initialize the environment
        self.env = gym.make("FetchReach-v4", render_mode=None)           #render_mode='human'
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
        self.epochs = 20000                           # n. of epochs during training. It is egual to the n. of episodes 
        self.batch_size = 256                         # batch size
        self.update_freq = 1                          # defines the network update frequency during training
        self.H = 50                                   # the horizon of one episode. Keep in mind the episode ends if we collect negative rewards for 50 consecutive steps
        self.rho = 0.05                               # defines the distance to the goal in which the reward is positive. Used in compute_reward
        self.gamma = 0.98                             # discount factor
        self.save_freq = 100                          # determines the after how much episodes the model is saved
        self.tau = 0.05                               # parameter for the soft update
        self.epsilon = 0.8                            # epsilon-greedy parameter
        self.EPOCH_EPSILON_DECAY_1 = 900
        self.EPOCH_EPSILON_DECAY_2 = 1500
        self.MIN_EPSILON =0.3

        # initialize the buffer
        self.replay_buffer = ReplayBuffer(capacity=1000, episode_horizon=self.H)

        # Initialize the logs for the evaluation
        self.training_logs = {
                "n_episodes": 0,
                "reward_per_episode": [],
                "actor_loss": [],
                "critic_loss": [],
                "training_time": [],
                "win_rate":[]
            }
        self.start_epoch = 1

        # Initialize the normalizers
        self.state_normalizer = Normalizer(self.state_dim)
        self.goal_normalizer = Normalizer(self.goal_dim)

    def act(self, observation):
        """ Determines the action(s) to take based on the given observation (state).
        This method is used only in main.py. It takes a single observation as input and returns the corresponding action as determined by the agent's policy.

        Parameters:
            observation: The current state of the environment.

        Returns:
            The selected action(s) based on the input observation.
        """
        self.target_actor.eval()
        
        # preprocessing
        normalized_state = self.state_normalizer.normalize(observation['observation'])
        normalized_goal = self.goal_normalizer.normalize(observation['desired_goal'])

        state =  torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0)
        desired_goal =  torch.tensor(normalized_goal, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            actions = np.array(self.target_actor(state, desired_goal))

        actions = np.hstack([actions, np.zeros((actions.shape[0], 1))])                # add 0 as the 4-th actions (in Reach task it is useless)
        return actions[0]

    def noisy_action(self, observation, noise_type ='Gaussian'):                       # noise_type = {Uniform, Gaussian, Ornstein-Uhlenbeck}
        """ This method takes a single observation as input and returns the action selected by the behavior policy.
            the behavioral policy is: beta = mu + N.
            Epsilon greedy strategy has been adopted to guarantee sufficient exploration.

        Parameters:
            observation: The current state of the environment.
            noise_type: The noise type to be added to the agent deterministic policy.
                        It can be {Uniform, Gaussian, Ornstein-Uhlenbeck}

        Returns:
            The action according to the behavior policy.
        """
        self.actor.eval()

        # preprocessing
        normalized_state = self.state_normalizer.normalize(observation['observation'])
        normalized_goal = self.goal_normalizer.normalize(observation['desired_goal'])

        state =  torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        desired_goal =  torch.tensor(normalized_goal, dtype=torch.float32).unsqueeze(0).to(self.device)
    
        with torch.no_grad():
            actions = np.array(self.actor(state, desired_goal).cpu())[0]

        # Noise types:
        if noise_type == 'Uniform':
            noise = np.random.uniform(-1, 1, size=actions.shape)                       # Uniform noise in the range [-1, 1]
        if noise_type == 'Gaussian':
            noise = 0.2 * np.random.randn(self.action_dim)                             # Gaussian noise with mean 0 and standard deviation 0.2  
        if noise_type == 'Ornstein-Uhlenbeck':                                         # not implemented
            pass
        actions += noise
        actions = np.clip(actions, -1, 1)

        # Epsilon-Greedy startegy to favor local exploration with prob self.epsilon
        actions += np.random.binomial(1, self.epsilon) * (
                np.random.uniform(-1,1,actions.shape[0]) - actions)  
        return actions
    
    def rollout(self):
        """This function implements the rollout phase that generates the training data.
            the selected behavior action is selected according to 'noisy_action' method
        """
        obs, _ = self.env.reset()                                 # obs contains the state and the goal
        done = False

        total_reward = 0
        for i in range(self.H):
            # Sample an action
            action = self.noisy_action(obs)
            action_4dim = np.hstack([action, np.zeros(1)])                             # append the last element as 0 to get a 4-dim action
            next_obs, reward, terminated, truncated, info = self.env.step(action_4dim)
            done = False if i != self.H -1 else True

            # Store transition
            self.replay_buffer.push(obs, action, next_obs, done)

            # Update the Normalizer
            self.state_normalizer.update(obs['observation'])
            self.goal_normalizer.update(obs['achieved_goal'])
            self.goal_normalizer.update(obs['desired_goal'])

            obs = next_obs
            total_reward += reward
        return total_reward                   # for the evaluation metric

    def compute_reward(self, next_obs, new_goal):
        """This function computes the reward:
                reward=0    if the distance of end-effector from the goal is < tau
                reward=-1   otherwise         
        """
        ee_position = next_obs[:-7]
        distance_form_the_goal = np.linalg.norm(np.array(ee_position) - np.array(new_goal))
        if distance_form_the_goal < self.rho:
            return 0
        else:
            return -1

    def train(self, lr_actor = 1e-3, lr_critic =1e-3, l2_lambda=0.5):  
        # Load the weights if model.pt exists
        if os.path.exists('model.pt'):
            self.load(load_for_training=True)                                           # load target actor and critic
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
            
            # Recompute mean and std for the Normalizers
            self.state_normalizer.recompute_stats()
            self.goal_normalizer.recompute_stats()
            
            # Epsilon_decay:
            if epoch > self.EPOCH_EPSILON_DECAY_1 and epoch < self.EPOCH_EPSILON_DECAY_2: self.epsilon= 0.7
            elif epoch >= self.EPOCH_EPSILON_DECAY_2: self.epsilon= self.MIN_EPSILON
                
            # Batch computation with update frequency
            if epoch % self.update_freq == 0:
                self.replay_buffer.set_probabilities()                        # set the probabilities in such a way that they sum up to 1 before sampling 

                # set the actor-critic in training mode 
                self.actor.train()
                self.critic.train()

                critic_loss = []
                actor_loss = []
    
                for k in range (self.batch_size):
                    episode_idx = self.replay_buffer.sample_episode()                                             # sample an episode
                    experience, new_goal, j, i = self.replay_buffer.sample_experience_and_goal(episode_idx)       # sample an experience with an hindsight goal according to the Future startegy
                                                                                                                  # here experience = (sj,aj,sj+1,delta_array, done)
                    # Show an example of sampled idx
                    # print(f'episode_idx: {episode_idx}, j: {j}, i: {i}')

                    next_obs = experience[2]['observation']
                    new_reward = self.compute_reward(next_obs,new_goal)

                    # Normalize the inputs
                    normalized_state = self.state_normalizer.normalize(experience[0]['observation'])
                    normalized_goal = self.goal_normalizer.normalize(new_goal)
                    normalized_next_state = self.state_normalizer.normalize(next_obs)

                    obs = torch.from_numpy(normalized_state).to(dtype=torch.float32).to(self.device)
                    action = torch.from_numpy(experience[1]).to(dtype=torch.float32).to(self.device)  
                    next_obs = torch.from_numpy(normalized_next_state).to(dtype=torch.float32).to(self.device)  
                    new_goal = torch.from_numpy(normalized_goal).to(dtype=torch.float32).to(self.device)  

                    # Compute importance sampling
                    w = self.replay_buffer.get_importance_sampling(episode_idx, j, i)
                    
                    with torch.no_grad():
                        target_mu = self.target_actor(next_obs, new_goal).detach()                                   # mu(s_(j+1) ||g_i)
                        next_Q_value = self.target_critic(next_obs, target_mu, new_goal).detach()                    # Q_target(s_(j+1), mu(s_(j+1) ||g_i) || g_i)
                    
                    target_return = new_reward + self.gamma * next_Q_value
                    target_return = torch.clamp(target_return, -1 / (1 - self.gamma), 0)                        
                    
                    Q_value = self.critic(obs, action, new_goal)                                                     # Q(s_j, a_j || g_i)

                    # Compute TD error
                    delta = (target_return - Q_value).squeeze(0)
                   
                    # Insert delta in buffer and Update the probabilities
                    self.replay_buffer.update_delta_and_probs(delta.detach().item(), episode_idx, j, i)

                    # Compute losses
                    critic_loss_k = w * delta ** 2

                    mu = self.actor(obs, new_goal)
                    actor_loss_k = - self.critic(obs, mu, new_goal)
                    actor_loss_k += l2_lambda * mu.pow(2).mean()

                    critic_loss.append(critic_loss_k)
                    actor_loss.append(actor_loss_k)

                # Update networks
                max_w = self.replay_buffer.get_max_w()

                # Normalize only critic loss 
                critic_loss = [loss / max_w for loss in critic_loss]

                # Mean the losses to accumulate the gradients
                actor_loss = torch.mean(torch.stack(actor_loss))
                critic_loss = torch.mean(torch.stack(critic_loss))

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

            # Save after 'save_freq' epochs
            if (epoch) % self.save_freq == 0:

                # Evaluate current win rate
                self.training_logs['win_rate'] = self.test()

                current_time = time.time()
                self.training_logs['training_time'].append((current_time-start_time)/3600)
                self.training_logs['n_episodes'] = epoch
                self.save()
                
                # Clear the logs buffer
                self.training_logs = {
                "n_episodes": 0,
                "reward_per_episode": [],
                "actor_loss": [],
                "critic_loss": [],
                "training_time":[],
                "win_rate":[]
            }
                start_time = current_time

    def soft_update(self, target_net, main_net):
        # Soft update for target net
        for target_param, local_param in zip(target_net.parameters(), main_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self):
        print('Saving the model and logs...')
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
            old_data['win_rate'].extend(self.training_logs['win_rate'])
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

        # Save the normalizers
        self.state_normalizer.save('state_normalizer.json')
        self.goal_normalizer.save('goal_normalizer.json')


    def load(self, load_for_training = False):
        # Load the model
        checkpoint = torch.load('model.pt', map_location=self.device)
        self.target_actor.load_state_dict(checkpoint['actor'])
        self.target_critic.load_state_dict(checkpoint['critic'])
        print("weights loaded for actor and critic!")

        # Load the current epoch
        file_path = 'training_logs.json'
        if os.path.exists(file_path):
            # Load normalizers
            self.state_normalizer.load('state_normalizer.json')
            self.goal_normalizer.load('goal_normalizer.json')
            with open(file_path, 'r') as f_load:
                logs = json.load(f_load)
                self.start_epoch = logs['n_episodes'] + 1
        
        # For training:
        if load_for_training:

            # Load the buffer
            if os.path.exists('buffer.pkl'):
                with open('buffer.pkl', 'rb') as file_load:
                    self.replay_buffer = pickle.load(file_load)
                    print('buffer loaded')
                    print('len buffer', len(self.replay_buffer.buffer))
                    print('max_delta', self.replay_buffer.max_delta)

    def plot_training_logs(self):
        """ Plot the evaluation metrics used for evaluating the training:
            -Total Reward per Episode during Training: used for evaluate exploration
            -Success rate per training time: tested with the test function
            -Actor Loss
            -Critic Loss
        """
        try:
            with open('training_logs.json', 'r') as f:
                logs = json.load(f)

            plot_logs(logs['reward_per_episode'], 
                      logs['win_rate'], 
                      logs['training_time'],
                      logs['actor_loss'],
                      logs['critic_loss'],
                      self.update_freq,
                      per_episodes_evaluation =50
                      )

        except FileNotFoundError:
            print("Error: File 'training_logs.json' not found.")

    def test(self, n_episodes=50):
        """ test the learned policy in 50 episodes"""
        print('testing...')
        win_rate = 0
        for episode in range(n_episodes):
            total_reward = 0
            s, _ = self.env.reset()
            for _ in range(50):
                action = self.act(s)
                s, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
            # compute if a win occurred in the episode
            if total_reward != -50:
                win_rate+=1

        win_rate /= n_episodes
        print(f'Win rate in {n_episodes} episodes:', win_rate)
        return [win_rate]

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
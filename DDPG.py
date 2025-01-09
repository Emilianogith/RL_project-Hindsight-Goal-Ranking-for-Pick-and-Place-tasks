import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os

import gymnasium as gym
import gymnasium_robotics

import numpy as np

from HGR import ReplayBuffer

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
        self.epochs = 5                             # n. of epochs during training. It is egual to the n. of episodes 
        self.batch_size = 4                         # batch size
        self.update_freq = 1                        # defines the network update frequency during training
        self.H = 50                                 # the horizon of one episode. Keep in mind the episode ends if we collect negative rewards for 50 consecutive steps
        self.rho = 0.05                             # defines the distance to the goal in which the reward is positive. Used in compute_reward
        self.gamma = 0.98                           # discount factor
        self.save_freq = 5                          # determines the after how much episodes the model is saved
        self.tau = 0.005                            # parameter for the soft update

        # initialize the buffer
        self.replay_buffer = ReplayBuffer(capacity=10, episode_horizon=self.H)#10000)

    def act(self, observation):
        """ used only in the main.py it gets one state as input and return the actions"""
        self.target_actor.eval()
        
        # preprocessing
        state =  torch.tensor(observation['observation'], dtype=torch.float32).unsqueeze(0)
        desired_goal =  torch.tensor(observation['desired_goal'], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            actions = np.array(self.target_actor(state, desired_goal))

        actions = np.hstack([actions, np.zeros((actions.shape[0], 1))])                # add 0 as the 4-th actions (in Reach task it is useless)
        print('actions: ', actions[0])
        return actions[0]

    def noisy_action(self, observation, noise_factor):
        self.actor.eval()

        # preprocessing
        state =  torch.tensor(observation['observation'], dtype=torch.float32).unsqueeze(0)
        desired_goal =  torch.tensor(observation['desired_goal'], dtype=torch.float32).unsqueeze(0)
    
        with torch.no_grad():
            actions = np.array(self.actor(state, desired_goal))[0]

        noise = np.random.uniform(-1, 1, size=actions.shape) * noise_factor            # Noise in the range [-1, 1] 
        actions += noise
        return actions
    
    def rollout(self, noise_factor=1, decay=0.999):
        obs, _ = self.env.reset()                                                      # obs contains the state and the goal
        done = False

        for _ in range(self.H):
            # Sample an action
            action = self.noisy_action(obs, noise_factor)
            action_4dim = np.hstack([action, np.zeros(1)])                             # append the last element as 0 to get a 4-dim action
            next_obs, reward, terminated, truncated, info = self.env.step(action_4dim)
            done = terminated or truncated

            noise_factor *= decay

            # Store transition
            self.replay_buffer.push(obs, action, next_obs, done)

            obs = next_obs

    def compute_reward(self, next_obs, new_goal):
        ee_position = next_obs[:-7]
        distance_form_the_goal = np.linalg.norm(np.array(ee_position) - np.array(new_goal))
        # print('distance_form_the_goal',distance_form_the_goal)
        if distance_form_the_goal < self.rho:
            return 0
        else:
            return -1

    def train(self):
        # Load the weights if model.pt exists
        if os.path.exists('model.pt'):
            self.load()                                                                 # load target actor and critic
            self.actor.load_state_dict(self.target_actor.state_dict())                  # copy the weights in the actor network
            self.critic.load_state_dict(self.target_critic.state_dict())                # copy the weights in the critic network
            self.target_actor.to(self.device)
            self.target_critic.to(self.device)
            self.actor.to(self.device)
            self.critic.to(self.device)

        # Initialize the optimizers for actor and critic
        actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Training loop
        for epoch in range(self.epochs):                # for each episode
            
            # Rollout phase
            self.rollout()

            # Batch computation with update frequency
            if epoch % self.update_freq == 0:
                self.replay_buffer.set_probabilities()                                                      # set the probabilities such that they sum up to 1 before sampling 
                
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

                    obs = torch.tensor(experience[0]['observation'], dtype=torch.float32).to(self.device)         
                    action = torch.tensor(experience[1],dtype=torch.float32).to(self.device)  
                    next_obs = torch.tensor(experience[2]['observation'],dtype=torch.float32).to(self.device)  
                    new_goal = torch.from_numpy(new_goal).to(dtype=torch.float32).to(self.device)  

                    new_reward = self.compute_reward(next_obs,new_goal) 

                    # Compute importance sampling
                    w = self.replay_buffer.get_importance_sampling(episode_idx, j, i)
                    importance_sampling.append(w)

                    target_mu = self.target_actor(next_obs, new_goal).detach()                                       # mu(s_(j+1) ||g_i)
                    Q_value = self.critic(obs, action, new_goal)                                                     # Q(s_j, a_j || g_i)
                    new_Q_value = self.target_critic(next_obs, target_mu, new_goal).detach()                         # Q_target(s_(j+1), mu(s_(j+1) ||g_i) || g_i)
                    
                    # Compute TD error
                    delta = (new_reward + self.gamma * new_Q_value - Q_value)[0]

                    # Insert delta in buffer and Update the probabilities
                    self.replay_buffer.update_delta_and_probs(delta.detach().item(), episode_idx, j, i)

                    # Compute losses
                    critic_loss_k = w * delta ** 2

                    mu = self.actor(obs, new_goal)
                    actor_loss_k = - self.critic(obs, mu, new_goal)  

                    critic_loss.append(critic_loss_k)
                    actor_loss.append(actor_loss_k)
                    # region work succesfully
                    # actor_loss += actor_loss_k
                    # critic_loss += critic_loss_k

                # actor_loss /= self.batch_size
                # critic_loss /= self.batch_size
                # actor_optimizer.zero_grad()
                # actor_loss.backward()  
                # actor_optimizer.step()
                # critic_optimizer.zero_grad()
                # critic_loss.backward()  
                # critic_optimizer.step()
                # endregion

                # Update networks
                actor_loss = [loss / max(importance_sampling) for loss in actor_loss]
                critic_loss = [loss / max(importance_sampling) for loss in critic_loss]

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

            print(f"Epoch {epoch + 1}/{self.epochs}, Actor loss: {actor_loss.item()}, , Critic loss: {critic_loss.item()}")
        
            # save after tot epochs
            if (epoch +1) % self.save_freq == 0:
                self.save()

    def soft_update(self, target_net, main_net):
        # Soft update for target net
        for target_param, local_param in zip(target_net.parameters(), main_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self):
        # salva anche le loss per plottare il gragico
        torch.save({
            'actor': self.target_actor.state_dict(),
            'critic': self.target_critic.state_dict()
                    }, 'model.pt')

    def load(self):
        checkpoint = torch.load('model.pt', map_location=self.device)
        self.target_actor.load_state_dict(checkpoint['actor'])
        self.target_critic.load_state_dict(checkpoint['critic'])
        print("weights loaded for actor and critic!")

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    


if __name__ == "__main__":
    agent = Policy()
    
    # FOR DEBUGGING
    agent.train()


    # accumulate the reward for evaluation
import numpy as np
import random

# OLD PREPROCESSING
# def preprocessing(self, states):
#         """preprocess the inputs expected as 2 types"""
#         if isinstance(states, list):
#             print('len', len(states))
#             observations = np.array([obs['observation'] for obs in states], dtype=np.float32)
#             # achieved_goal = np.array([obs['achieved_goal'] for obs in states], dtype=np.float32) # inutile
#             desired_goal = np.array([obs['desired_goal'] for obs in states], dtype=np.float32)

#             observations = torch.from_numpy(observations)
#             # achieved_goal = torch.from_numpy(achieved_goal) inutile
#             desired_goal = torch.from_numpy(desired_goal)

#         else:
#             observations =  torch.tensor(states['observation'], dtype=torch.float32).unsqueeze(0)
#             # achieved_goal =  torch.tensor(states['achieved_goal'], dtype=torch.float32).unsqueeze(0) # inutile
#             desired_goal =  torch.tensor(states['desired_goal'], dtype=torch.float32).unsqueeze(0)
        
#         # print('observations shape:', observations.shape)
#         # print(observations)
#         return observations, desired_goal#, achieved_goal


# OLD CLASS ACTOR_CRITIC

# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, goal_dim):
#         super(ActorCritic, self).__init__()
#         # Actor
#         self.actor = nn.Sequential(
#             nn.Linear(state_dim + goal_dim, 256), nn.ReLU(),                            # try LeakyReLU 
#             nn.Linear(256, 256), nn.ReLU(),
#             nn.Linear(256, action_dim), nn.Tanh()
#         )
#         # Critic
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim + action_dim + goal_dim, 256), nn.ReLU(),
#             nn.Linear(256, 256), nn.ReLU(),
#             nn.Linear(256, 1)
#         )
#         self.initialize_layers()

#     def initialize_layers(self):
#         for layer in self.actor, self.critic:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
#                 nn.init.zeros_(layer.bias)

#     def get_critic(self, state, action, goal):
#         return self.critic(torch.cat([state, action, goal], dim=-1))

#     def get_actor(self, state, goal):
#         return self.actor(torch.cat([state, goal], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity, episode_horizon):
        self.capacity = capacity
        self.H = episode_horizon
        self.buffer = [[]]
        #self.position = 0
        self.last_done = False

        self.episode_prioritization = []
        self.goal_prioritization = [[]]

        self.alpha=0.09
        self.alpha_prime=0.09

        self.epsilon = 0.001


    # def push(self, episode, obs, action, next_obs, done):
    #     delta = self.max_priority()
    #     if len(self.buffer) < self.capacity:
    #         if episode > len(self.buffer) -1:
    #             self.buffer.append([])
    #             self.position = 0                   #modifica il rimpiazzo del buffer
    #         self.buffer[episode].append(None)
    #     self.buffer[episode][self.position] = (obs, action, next_obs, delta, done)
    #     self.position = (self.position + 1) % self.capacity        # to replace the oldest experience. (have i to replace according to prioritization????)
    
    def push(self, obs, action, next_obs, done):
        delta_array = self.max_priority()

        if self.last_done:
            self.buffer.append([])
            self.last_done = False

        self.buffer[-1].append((obs, action, next_obs, delta_array, done))

        if done:
            self.last_done = True
        if len(self.buffer) > self.capacity:
            print('poppato')
            self.pop_buffer()

    def pop_buffer(self):
        self.buffer.pop(0)          # to replace the oldest experience. (have i to replace according to prioritization????)

    def max_priority(self):
        if self.buffer == [[]]:
            max_delta = np.zeros(self.H)

        else:       
            max_delta = 0                                
            for ep in range(len(self.buffer)):                      # cycle for all the episodes
                for experience in self.buffer[ep]:                  # cycle for the experiences in an episode
                    for delta_ji in experience[3]:                  # cycle for all i in delta_ji 
                        if delta_ji > max_delta:
                            max_delta = delta_ji

            remaining_length = self.H - len(self.buffer[-1])
            if remaining_length == 0: print('hjhusuwi', remaining_length)
            max_delta = np.ones(remaining_length) * max_delta                                    #max(experience[3] for experience in self.buffer[ep])
            print(max_delta)
            # max_delta = np.zeros(remaining_length)  if remaining_length != 0 else None

        return max_delta #+ np.ones(max_delta.shape[0]) * self.epsilon                          # small positive number to prevent from 0 probability
    
    def compute_episode_delta(self, idx):
        episode_delta = 0
        episode = self.buffer[idx]
        K = 0
        for experience in episode:
            delta_array = experience[3]
            episode_delta += np.abs(np.sum(delta_array))
            K += delta_array.shape[0]
        # print('possible experience-goal combinations',K)        # K = 1275= H*(H+1)/2
        return episode_delta / K

        # H = len(self.buffer[-1])
        # K = int(H*(H+1)/2)
        # print('K',K)
        # for k in range(K):                              # calcola le possibili combinazioni 
        #     episode_delta += np.abs(self.buffer[-1][k][3]) # get delta
        # return episode_delta / K
    
    def sample(self, batch_size):                    # RIFAI
        batch = random.sample(self.buffer, batch_size)
        obs, action, next_obs, done = map(np.stack, zip(*batch))
        return obs, action, next_obs, done
    
    # def sample_batch(self, batch_size):
    #     batch = []
    #     for k in range (batch_size):
    #         # Episode prioritization: Sample n-th episode for replaying based on P(n)
    #         indices = range(len(self.buffer))
            
    #         if self.episode_prioritization == []:
    #             # uniform probability at the first iteration
    #             selected_episode = random.choices(indices, k=1)         #return the index
    #             exp_indices = range(len(self.buffer[selected_episode]))
    #             selected_experience = random.choices(exp_indices, k=1) 
            
    #         else:
    #             selected_episode = random.choices(indices, weights=self.episode_prioritization, k=1)
    #             exp_indices = range(len(self.buffer[selected_episode]))
    #             selected_experience = random.choices(exp_indices, weights=self.goal_prioritization[selected_episode], k=1) 
            
    def sample_episode(self):
        # Episode prioritization: Sample n-th episode for replaying based on P(n)
        indices = range(len(self.buffer))
        delta_episodes = np.array([])
        for idx in indices:
            delta_n = self.compute_episode_delta(idx)
            delta_episodes = np.append(delta_episodes, delta_n)

        episode_probabilities = np.where(delta_episodes == 0, 1, delta_episodes ** self.alpha)
        episode_probabilities /= np.sum(episode_probabilities)
        
        selected_episode_idx = random.choices(indices, weights=episode_probabilities, k=1)
        return selected_episode_idx[0] # the index
    
    def sample_experience_and_goal(self, selected_episode_idx):
        # Goal prioritization: Sample j-th experience and i-th goal based on P'(j,1)
        # indices = range(len(self.buffer[selected_episode_idx]))
        selected_episode = self.buffer[selected_episode_idx]

        probabilities= np.zeros((self.H-1, self.H))
        for j,experience in enumerate(selected_episode[:-1]):

            print(selected_episode[j])
            for i,goal in enumerate(selected_episode[j+1:]):
                delta_ji=np.abs(selected_episode[j][3][i])
                probabilities[j][i]= delta_ji ** self.alpha_prime if delta_ji !=0 else 1        # access to delta

        # normalization_factor=np.sum(np.sum( probabilities, axis=0), axis=0)
        # probabilities/= np.sum(normalization_factor)

        #print(np.sum(np.sum( probabilities, axis=0), axis=0))

        # Flatten della matrice di probabilità
        flattened_probabilities = probabilities.flatten()
        flattened_probabilities /= np.sum(flattened_probabilities)

        # Campionamento di un indice lineare
        index = np.random.choice(len(flattened_probabilities), p=flattened_probabilities)

        # Ricostruzione degli indici j, i
        j, i = np.unravel_index(index, probabilities.shape)

        print(f"Campionato indice (j, i): ({j}, {i})")
            
        new_goal = selected_episode[j+i][0]['achieved_goal']
        print('finish',selected_episode[j], new_goal)
        return selected_episode[j], new_goal



        # experience_idx = random.choices(indices, k=1) 

        # if self.episode_prioritization == []:
        #     # uniform probability at the first iteration
        #     selected_goal = random.choices(indices, k=1)         #return the index
        #     new_goal = self.buffer[selected_episode_idx][selected_goal]['achieved_goal']
        # else:
        #     new_goal = self.compute_new_goal(indices, selected_episode_idx)         #RIFAI
            
        # return self.buffer[selected_episode_idx][experience_idx], new_goal
    
    def compute_new_goal(self, indices, selected_episode_idx):
        selected_goal = random.choices(indices, weights=self.goal_prioritization[selected_episode_idx], k=1)
        goal = self.buffer[selected_episode_idx][selected_goal]['achieved_goal']
        return goal

    def __len__(self):
        buffer_size = 0
        for ep in range(len(self.buffer)):
            buffer_size += len(self.buffer[ep])
        return buffer_size
    

# class ReplayBuffer:
#     def __init__(self, gamma=0.99, k_future=4, hgr=True):
#         """
#         Hindsight Experience Replay with Hindsight Goal Ranking.
#         Args:
#             env: Gym-like environment.
#             buffer_size: Maximum size of the replay buffer.
#             gamma: Discount factor for rewards.
#             k_future: Number of future goals sampled for HER.
#             hgr: Whether to use Hindsight Goal Ranking.
#         """

#         self.gamma = gamma
#         self.k_future = k_future
#         self.hgr = hgr

#         # Replay buffer
#         self.buffer = []

#     def store_transition(self, obs, action, next_obs, done):
#         """
#         Store a transition in the replay buffer.
#         """
#         transition = {
#             'obs': obs,
#             'action': action,
#             'next_obs': next_obs,
#             'done': done
#             #'priority': ?????
#         }
        
#         self.buffer.append(transition)

#     def __len__(self):
#         return len(self.buffer)
           

#     def sample(self, batch_size):
#         """
#         Sample a batch of transitions with HER relabeling and HGR ranking.
#         """
#         batch = random.sample(self.buffer, batch_size)

#         obs, actions, rewards, next_obs, dones, goals = [], [], [], [], [], []
#         for transition in batch:
#             obs.append(transition['obs'])
#             actions.append(transition['action'])
#             rewards.append(transition['reward'])
#             next_obs.append(transition['next_obs'])
#             dones.append(transition['done'])
#             goals.append(transition['goal'])

#             # HER relabeling: Sample future goals
#             future_goals = self._sample_future_goals()
#             for fg in future_goals:
#                 relabeled_reward = self._compute_reward(next_obs[-1], fg)
#                 self.store_transition(
#                     transition['obs'], transition['action'], relabeled_reward,
#                     transition['next_obs'], transition['done'], fg
#                 )

#         return np.array(obs), np.array(actions), np.array(rewards), np.array(next_obs), np.array(dones), np.array(goals)

#     def _sample_future_goals(self):
#         """
#         Sample k_future future goals from the replay buffer.
#         """
#         indices = np.random.choice(len(self.buffer), self.k_future, replace=False)
#         return [self.buffer[i]['goal'] for i in indices]

#     def _compute_reward(self, achieved_goal, desired_goal):
#         """
#         Compute the reward based on the goal achievement measure (GAM).
#         """
#         return 1.0 if np.linalg.norm(achieved_goal - desired_goal) < self.env.goal_threshold else 0.0


class GoalRanker:
    def __init__(self, buffer):
        """
        Initialize the Goal Ranker with a replay buffer.
        Args:
            buffer: Replay buffer with stored transitions.
        """
        self.buffer = buffer

    def rank_goals(self, achieved_goals, desired_goals):
        """
        Rank goals based on the Goal Achievement Measure (GAM).
        Args:
            achieved_goals: List of achieved goals.
            desired_goals: List of desired goals.
        Returns:
            Ranked goals based on contribution.
        """
        gam_scores = [
            np.linalg.norm(ag - dg) for ag, dg in zip(achieved_goals, desired_goals)
        ]
        ranked_indices = np.argsort(gam_scores)  # Lowest GAM score is the most useful
        return [desired_goals[i] for i in ranked_indices]

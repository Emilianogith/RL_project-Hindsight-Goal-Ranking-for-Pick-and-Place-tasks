import numpy as np
import random


class ReplayBuffer:
    def __init__(self, capacity, episode_horizon):
        """
        Buffer structure:

          |              ------episode-----               |
        [ [ [obs,action,next_obs,delta_array,done], [],.. ] , delta_episode ] , [] ]

        """
        self.capacity = capacity
        self.H = episode_horizon
        
        self.buffer = []
        self.terminated_last_episode = True     # initialized at True

        self.episode_prioritization_probs = []
        self.goal_prioritization_probs = []

        # Hyperparameters
        self.alpha=0.6
        self.alpha_prime=0.6
        self.beta=0.4
        self.beta_prime=0.4
        self.epsilon = 0.001

        self.max_delta = self.epsilon           # initialized as self.epsilon, AGGIORNA QUANDO IMMETTI DELTA_JI

    def push(self, obs, action, next_obs, done):

        if self.terminated_last_episode:
            self.buffer.append([])
            self.terminated_last_episode = False
            remaining_length = self.H - 1
        else:
            remaining_length = self.H - len(self.buffer[-1]) - 1

        delta_array = [self.max_delta for i in range(remaining_length)]
        self.buffer[-1].append((obs, action, next_obs, delta_array, done))

        if done:
            self.terminated_last_episode = True
            self.buffer[-1] = [self.buffer[-1], self.compute_episode_delta(episode = self.buffer[-1])]
        if len(self.buffer) > self.capacity:
            print('poppato')
            self.pop_buffer()

    def pop_buffer(self):
        self.buffer.pop(0)          # to replace the oldest experience. (have i to replace according to prioritization????)

    def compute_episode_delta(self, episode):
        episode_delta = 0
        K = 0
        for experience in episode:
            delta_array = experience[3]
            episode_delta += np.abs(np.sum(delta_array))
            K += len(delta_array)
        # print('possible experience-goal combinations',K)        # K = 1225= H*(H-1)/2
        return episode_delta / K

       
    def set_probabilities(self):
        # set goal_prioritization_probs
        self.episode_prioritization_probs = []
        self.goal_prioritization_probs = []
        episode_norm_factor =0
        for idx,episode in enumerate(self.buffer):
            self.goal_prioritization_probs.append([])
            goal_norm_factor = 0
   
            for j,experience in enumerate(episode[0][:-1]):      # for j = 1,.. H-1
                delta_array = experience[3]
                self.goal_prioritization_probs[idx].append([])
                for delta_ji in delta_array:       # for i = j+1,...H
                    if delta_ji == 0: raise ValueError("IMPOSSIBLE: delta_ji egual to 0")
                    delta_ji = np.abs(delta_ji) ** self.alpha_prime 
                    self.goal_prioritization_probs[idx][j].append(delta_ji)
                    goal_norm_factor += delta_ji

            for j in range(len(self.goal_prioritization_probs[idx])):
                for i,value in enumerate(self.goal_prioritization_probs[idx][j]):
                    self.goal_prioritization_probs[idx][j][i] =self.goal_prioritization_probs[idx][j][i]/goal_norm_factor

            # set episode_prioritization_probs
            if episode[1] == 0: raise ValueError("IMPOSSIBLE: delta_episode egual to 0")
            delta_episode = np.abs(episode[1]) ** self.alpha
            self.episode_prioritization_probs.append(delta_episode)
            episode_norm_factor += delta_episode
        
        self.episode_prioritization_probs /= episode_norm_factor    
 
    def sample_episode(self):
        # Episode prioritization: Sample n-th episode for replaying based on P(n)
        indices = range(len(self.episode_prioritization_probs))
        
        selected_episode_idx = random.choices(indices, weights=self.episode_prioritization_probs, k=1)
        return selected_episode_idx[0] # the index
    
    def sample_experience_and_goal(self, episode_idx):
        # Goal prioritization: Sample j-th experience and i-th goal based on P'(j,1)
        flattened_probabilities = []
        for prob in self.goal_prioritization_probs[episode_idx]:
            flattened_probabilities.extend(prob)

        sampled_index = np.random.choice(len(flattened_probabilities), p=flattened_probabilities)

        count=0
        for j, exp in enumerate(self.buffer[episode_idx][0][:-1]):
            for i, future_experience in enumerate(self.buffer[episode_idx][0][j+1:]):
                if count==sampled_index:
                    new_goal = future_experience[0]['achieved_goal']
                    # print('new goal', new_goal)
                    # print('presumed new goal', self.buffer[episode_idx][0][j+1+i])              # check: it works!
                    return exp, new_goal, j, i                                          
                count+=1

    def update_delta_and_probs(self, delta, episode, j,i):
        delta = np.abs(delta)
        self.buffer[episode][0][j][3][i] = delta

        if delta > self.max_delta: 
            self.max_delta = delta + self.epsilon
        
        self.buffer[episode][1] = self.compute_episode_delta(episode= self.buffer[episode][0])
        self.update_probabilities(episode)

    def update_probabilities(self,episode_idx):
        # update episode probabilities
        episode_norm_factor =0
        for idx,episode in enumerate(self.buffer):
            if episode[1] == 0: raise ValueError("IMPOSSIBLE: delta_episode egual to 0")
            delta_episode = np.abs(episode[1]) ** self.alpha
            self.episode_prioritization_probs[idx] = delta_episode
            episode_norm_factor += delta_episode
        
        self.episode_prioritization_probs /= episode_norm_factor   

        # update goal probabilities
        goal_norm_factor = 0
        for j,experience in enumerate(self.buffer[episode_idx][0][:-1]):      # for j = 1,.. H-1
            delta_array = experience[3]
            for i,delta_ji in enumerate(delta_array):       # for i = j+1,...H
                if delta_ji == 0: raise ValueError("IMPOSSIBLE: delta_ji egual to 0")
                delta_ji = np.abs(delta_ji) ** self.alpha_prime 
                self.goal_prioritization_probs[episode_idx][j][i] = delta_ji
                goal_norm_factor += delta_ji

        for j in range(len(self.goal_prioritization_probs[episode_idx])):
            for i,value in enumerate(self.goal_prioritization_probs[episode_idx][j]):
                self.goal_prioritization_probs[episode_idx][j][i] =self.goal_prioritization_probs[episode_idx][j][i]/goal_norm_factor

    def get_importance_sampling(self,n,j,i):
        w_n = (1/len(self.buffer) * 1/self.episode_prioritization_probs[n]) ** self.beta
        w_ji = (2/(self.H*(self.H-1))  * 1/self.goal_prioritization_probs[n][j][i]) ** self.beta_prime
        w = w_ji * w_n
        return w
    
    def __len__(self):
        buffer_size = 0
        for ep in range(len(self.buffer)):
            buffer_size += len(self.buffer[ep])
        return buffer_size
    
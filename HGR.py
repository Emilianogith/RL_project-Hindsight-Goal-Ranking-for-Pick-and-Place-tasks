import numpy as np
import random


class ReplayBuffer:
    def __init__(self, capacity, episode_horizon):
        """
        Buffer structure:

            |-------------------episode-------------------|                 
            |-------------experience-------------|        
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
        self.epsilon = 0.001                    # small positicve constant to be addedd to delta_ji in order to avoid 0 probability 

        self.max_delta = self.epsilon           # initialized as self.epsilon
        self.max_w = 0

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
            episode_delta += np.sum(np.abs(delta_array))
            K += len(delta_array)
        #print('possible experience-goal combinations',K)        # K = 1225= H*(H-1)/2
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
                for delta_ji in delta_array:                     # for i = j+1,...H
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
        # Goal prioritization: Sample j-th experience and i-th goal based on P'(j,i)
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

    def sample_random_experience_and_future_goal(self, episode_idx):
        """i have reimplemented the ""sample_experience_and_goal"" function that given the sampled episode sample an experience randomly 
        and a future goal according to the future startegy
        
        PROBLEMS WITH THE NORMALIZATION FACTOR!!!!!!!"""
        # # Goal prioritization: Sample randomly the j-th experience and sample the i-th goal based on P'(j,i)
        # episode_probs = self.goal_prioritization_probs[episode_idx]
        # j =  np.random.choice(len(self.buffer[episode_idx][0][:-1]))

        # print('j with uniform probs', j)
        # #future_probs = episode_probs[j+1:]
        # #print(len(future_probs))

        # # PROBLEMS WITH THE NORMALIZATION FACTOR!!!!!!!
        # # update goal probabilities
        # goal_norm_factor = 0
        # for exp,experience in enumerate(self.buffer[episode_idx][0][:-1]):      # for exp = 1,.. H-1
        #     delta_array = experience[3]
        #     for i,delta_ji in enumerate(delta_array):                         # for i = exp+1,...H
        #         if delta_ji == 0: raise ValueError("IMPOSSIBLE: delta_ji egual to 0")
        #         delta_ji = np.abs(delta_ji) ** self.alpha_prime 
        #         if exp == j:
        #             print(len(delta_array))
        #             self.goal_prioritization_probs[episode_idx][exp][i] = delta_ji
        #         goal_norm_factor += delta_ji

        # for exp in range(len(self.goal_prioritization_probs[episode_idx])):
        #     for i,value in enumerate(self.goal_prioritization_probs[episode_idx][exp]):
        #         self.goal_prioritization_probs[episode_idx][exp][i] =self.goal_prioritization_probs[episode_idx][exp][i]/goal_norm_factor

        # future_probs = self.goal_prioritization_probs[episode_idx][j]
        # print(future_probs)
        # print('len future probs', len(future_probs))

        # i = np.random.choice(len(future_probs), p=future_probs)
        # print('i based on HGR', i)

        # experience = self.buffer[episode_idx][0][j]
        # new_goal = self.buffer[episode_idx][0][j+1+i]
        # print('new goal sampled',new_goal)
        # return experience, new_goal, j, i
        pass

    def update_delta_and_probs(self, delta, episode, j,i):
        delta = np.abs(delta)
        self.buffer[episode][0][j][3][i] = delta

        if delta > self.max_delta: 
            self.max_delta = delta + self.epsilon                                                 # small positicve constant to be addedd to delta_ji in order to avoid 0 probability
        
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
            for i,delta_ji in enumerate(delta_array):                         # for i = j+1,...H
                if delta_ji == 0: raise ValueError("IMPOSSIBLE: delta_ji egual to 0")
                delta_ji = np.abs(delta_ji) ** self.alpha_prime 
                self.goal_prioritization_probs[episode_idx][j][i] = delta_ji
                goal_norm_factor += delta_ji

        # Normalize probs
        for j in range(len(self.goal_prioritization_probs[episode_idx])):
            for i,value in enumerate(self.goal_prioritization_probs[episode_idx][j]):
                self.goal_prioritization_probs[episode_idx][j][i] =self.goal_prioritization_probs[episode_idx][j][i]/goal_norm_factor

    def get_importance_sampling(self,n,j,i):
        w_n = (1/len(self.buffer) * 1/self.episode_prioritization_probs[n]) ** self.beta
        w_ji = (2/(self.H*(self.H-1))  * 1/self.goal_prioritization_probs[n][j][i]) ** self.beta_prime
        w = w_ji * w_n
        if w > self.max_w: self.max_w = w
        return w
    
    def get_max_w(self):
        return self.max_w
    
    def __len__(self):
        buffer_size = 0
        for ep in range(len(self.buffer)):
            buffer_size += len(self.buffer[ep])
        return buffer_size
    
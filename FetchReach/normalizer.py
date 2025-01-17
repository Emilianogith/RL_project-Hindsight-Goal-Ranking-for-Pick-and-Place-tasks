import numpy as np
import json

class Normalizer:
    def __init__(self, size, eps=1e-2):
        self.size = size
        self.eps = eps

        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)

        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.zeros(1, np.float32)                              

        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)

        self.local_sum += v.sum(axis=0)
        self.local_sumsq += (np.square(v)).sum(axis=0)
        self.local_count[0] += v.shape[0]

    def recompute_stats(self):
        local_count = self.local_count.copy()
        local_sum = self.local_sum.copy()
        local_sumsq = self.local_sumsq.copy()
        # reset
        self.local_count[...] = 0
        self.local_sum[...] = 0
        self.local_sumsq[...] = 0
        # update the total sums
        self.total_sum += local_sum
        self.total_sumsq += local_sumsq
        self.total_count += local_count
        
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(
            self.total_sum / self.total_count)))

    # normalize the observation
    def normalize(self, v):
        return (v - self.mean) / self.std
    
    def to_dict(self):
        return {
            'local_sum': self.local_sum.tolist(),
            'local_sumsq': self.local_sumsq.tolist(),
            'local_count': self.local_count.tolist(),
            'total_sum': self.total_sum.tolist(),
            'total_sumsq': self.total_sumsq.tolist(),
            'total_count': self.total_count.tolist(),
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'eps': self.eps,
        }
    
    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f)

    def load(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.local_sum = np.array(data['local_sum'], dtype=np.float32)
            self.local_sumsq = np.array(data['local_sumsq'], dtype=np.float32)
            self.local_count = np.array(data['local_count'], dtype=np.float32)
            self.total_sum = np.array(data['total_sum'], dtype=np.float32)
            self.total_sumsq = np.array(data['total_sumsq'], dtype=np.float32)
            self.total_count = np.array(data['total_count'], dtype=np.float32)
            self.mean = np.array(data['mean'], dtype=np.float32)
            self.std = np.array(data['std'], dtype=np.float32)
            self.eps = data['eps']
import numpy as np
import itertools

def rnd_int(N=100000):
    return np.random.randint(N)

def dict_combiner(mydict):
    if mydict:
        keys, values = zip(*mydict.items())
        experiment_list = [dict(zip(keys, v))
                           for v in itertools.product(*values)]
    else:
        experiment_list = [{}]
    return experiment_list

class MaxMinNormalizer(object):  
    def __init__(self, x, eps=1e-5):  
        super(MaxMinNormalizer, self).__init__()  
  
        self.max = np.max(x, 0)  
        self.min = np.min(x, 0) 
        self.range = self.max - self.min
        self.eps = eps  
  
    def encode(self, x):  
        return (x - self.min) / (self.range + self.eps)  
  
    def decode(self, x):  
        return self.min + x * (self.range + self.eps)

class UnitGaussianNormalizer(object):  
    def __init__(self, x, eps=1e-5):  
        super(UnitGaussianNormalizer, self).__init__()  
  
        self.mean = np.mean(x, 0)  
        self.std = np.std(x, 0) 
        self.eps = eps  
  
    def encode(self, x):  
        return (x - self.mean) / (self.std + self.eps)  
  
    def decode(self, x):  
        return (x * (self.std + self.eps)) + self.mean  

class InactiveNormalizer(object):  
    def __init__(self, x, eps=1e-5):  
        super(InactiveNormalizer, self).__init__()  
    
    def encode(self, x):  
        return x 
  
    def decode(self, x):  
        return x

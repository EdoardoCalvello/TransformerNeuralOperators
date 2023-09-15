import numpy as np
import itertools
from datetime import datetime

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

def subsample_and_flatten(matrix, stride):
    """
    Subsamples a matrix and flattens it into a 1D array.
    The subsampling is done by extracting the first and last rows and columns,
    and then extracting the interior elements based on the stride.
    The extracted elements are then flattened into a 1D array.
    The function returns the indices of the extracted elements and the flattened array.
    """
    # matrix: a 3D numpy array of shape (N, rows, cols)
    # stride: an integer representing the stride

    # Get the dimensions of the input matrix
    N, rows, cols = matrix.shape

    # Create a list to store the indices of the elements to be extracted
    indices = []

    # Add indices for the first row (left to right)
    indices.extend((0, j) for j in range(cols))

    # Add indices for the last row (left to right)
    if rows > 1:
        indices.extend((rows - 1, j) for j in range(cols))

    # Add indices for the first column (top to bottom, excluding corners)
    if rows > 2:
        indices.extend((i, 0) for i in range(1, rows - 1))

    # Add indices for the last column (top to bottom, excluding corners)
    if rows > 2:
        indices.extend((i, cols - 1) for i in range(1, rows - 1))

    # print(indices)
    # Generate indices for the interior elements based on the stride
    counter = 0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if counter % stride == 0:
                indices.append((i, j))
                counter = 0
            counter += 1

    # sort the indices
    indices.sort(key=lambda x: (x[0], x[1]))

    # Extract the elements from the matrix using the sorted indices
    result = [matrix[:, i, j] for i, j in indices]

    return np.array(indices).astype('float32'), np.array(result).T.astype('float32')

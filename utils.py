import numpy as np
import itertools
from datetime import datetime
from numpy.fft import rfft2

import itertools

def dict_combiner(mydict):
    """
    Combines the values of a dictionary into a list of dictionaries,
    where each dictionary represents a combination of the values.

    Args:
        mydict (dict): The input dictionary containing keys and lists of values.

    Returns:
        list: A list of dictionaries, where each dictionary represents a combination
              of the values from the input dictionary.

    Example:
        >>> mydict = {'A': [1, 2], 'B': [3, 4]}
        >>> dict_combiner(mydict)
        [{'A': 1, 'B': 3}, {'A': 1, 'B': 4}, {'A': 2, 'B': 3}, {'A': 2, 'B': 4}]
    """
    if mydict:
        keys, values = zip(*mydict.items())
        experiment_list = [dict(zip(keys, v))
                           for v in itertools.product(*values)]
    else:
        experiment_list = [{}]
    return experiment_list

class MaxMinNormalizer(object):
    """
    A class for performing max-min normalization on a given input array.

    Parameters:
    - x: Input array to be normalized.
    - eps: Small value added to the denominator to avoid division by zero.

    Methods:
    - encode(x): Normalize the input array using max-min normalization.
    - decode(x): Denormalize the input array using max-min normalization.
    """

    def __init__(self, x, eps=1e-5):
        super(MaxMinNormalizer, self).__init__()

        self.max = np.max(x, 0)
        self.min = np.min(x, 0)
        self.range = self.max - self.min
        self.eps = eps

    def encode(self, x):
        """
        Normalize the input array using max-min normalization.

        Parameters:
        - x: Input array to be normalized.

        Returns:
        - Normalized array.
        """
        return (x - self.min) / (self.range + self.eps)

    def decode(self, x):
        """
        Denormalize the input array using max-min normalization.

        Parameters:
        - x: Input array to be denormalized.

        Returns:
        - Denormalized array.
        """
        return self.min + x * (self.range + self.eps)

class UnitGaussianNormalizer(object):
    """
    A class for normalizing data using unit Gaussian normalization.

    Attributes:
        mean (numpy.ndarray): The mean values of the input data.
        std (numpy.ndarray): The standard deviation values of the input data.
        eps (float): A small value added to the denominator to avoid division by zero.

    Methods:
        encode(x): Normalize the input data using unit Gaussian normalization.
        decode(x): Denormalize the input data using unit Gaussian normalization.
    """

    def __init__(self, x, eps=1e-5):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = np.mean(x, 0)
        self.std = np.std(x, 0)
        self.eps = eps

    def encode(self, x):
        """
        Normalize the input data using unit Gaussian normalization.

        Args:
            x (numpy.ndarray): The input data to be normalized.

        Returns:
            numpy.ndarray: The normalized data.
        """
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        """
        Denormalize the input data using unit Gaussian normalization.

        Args:
            x (numpy.ndarray): The normalized data to be denormalized.

        Returns:
            numpy.ndarray: The denormalized data.
        """
        return (x * (self.std + self.eps)) + self.mean

class InactiveNormalizer(object):
    """
    A class for normalizing and denormalizing data using an inactive approach.
    """

    def __init__(self, x, eps=1e-5):
        super(InactiveNormalizer, self).__init__()

    def encode(self, x):
        """
        Encodes the input data.

        Args:
            x: The input data to be encoded.

        Returns:
            The encoded data.
        """
        return x

    def decode(self, x):
        """
        Decodes the input data.

        Args:
            x: The input data to be decoded.

        Returns:
            The decoded data.
        """
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

def patch_coords(matrix, stride):
    """
    Generate coordinates for each element in the matrix.

    Args:
        matrix (ndarray): Input matrix.
        stride (int): Stride value.

    Returns:
        ndarray: Array of coordinates for each element in the matrix.
    """
    N, rows, cols = matrix.shape

    nx = np.linspace(0,1,num=cols)
    ny = np.linspace(0,1,num=rows)

    coords = np.array(np.meshgrid(nx,ny))
    
    return coords


def fourier_coords(matrix, stride):

    N, rows, cols = matrix.shape
    
    rows = rows
    cols = cols
    #freq_rows = np.fft.fftfreq(rows)
    #freq_cols = np.fft.fftfreq(cols)

    # concatenate two 1d numpy arrays that are linspace of the same length
    '''
    freq_rows = 1/np.concatenate((np.linspace(cols//2,1,num=cols//2),np.linspace(-1,-cols//2,num=cols//2)))
    freq_cols = 1/np.concatenate((np.linspace(rows//2,1,num=rows//2),np.linspace(-1,-rows//2,num=rows//2)))
    '''

    freq_rows = 1/np.linspace(1,cols,num=cols)
    freq_cols = 1/np.linspace(1,rows,num=rows)

    coords = np.array(np.meshgrid(freq_rows,freq_cols))

    return coords

def FourierNormalizer(matrix):
    # matrix: a 3D numpy array of shape (N, rows, cols)

    N, rows, cols = matrix.shape
    # matrix: a 3D numpy array of shape (N, rows, cols)
    fft = np.fft.rfft2(matrix, axes=(1, 2))[:,:,:cols//2]
    mean_real = np.mean(fft.real.reshape(-1,1),0)[0]#*np.ones(fft.real.shape[1:])
    mean_complex = np.mean(fft.imag.reshape(-1,1),0)[0]#*np.ones(fft.real.shape[1:])
    std_real = np.std(fft.real.reshape(-1,1),0)[0]+ 1e-5#*np.ones(fft.real.shape[1:])
    std_complex = np.std(fft.imag.reshape(-1,1),0)[0]+ 1e-5#*np.ones(fft.real.shape[1:])


    return [mean_real, mean_complex, std_real, std_complex]

def fourier_transformation(matrix):
        
        N, rows, cols = matrix.shape

        #check this whole thing
        x = rfft2(matrix, axes=(-2,-1), norm="ortho")[...,:-1] # (N, rows, cols//2)
        '''
        normalizer[0] = normalizer[0].unsqueeze(1)
        normalizer[1] = normalizer[1].unsqueeze(1)
        normalizer[2] = normalizer[2].unsqueeze(1)
        normalizer[3] = normalizer[3].unsqueeze(1)
        #import pdb; pdb.set_trace()
        x_real = torch.div(x.real-normalizer[0],normalizer[2]).to(torch.float32)
        x_imag = torch.div(x.real-normalizer[1],normalizer[3]).to(torch.float32)
        '''


        #x_real = x.real.to(torch.float32)
        #x_imag = x.imag.to(torch.float32)
        #x = torch.cat((x_real, x_imag), dim=1) # (batch_size, 2*(input_dim+domain_dim), rows, cols//2+1)

        return x

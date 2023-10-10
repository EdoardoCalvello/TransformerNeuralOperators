import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
import pytorch_lightning as pl
import scipy.io
from sklearn.model_selection import train_test_split
from utils import InactiveNormalizer, UnitGaussianNormalizer, MaxMinNormalizer
from utils import subsample_and_flatten

from pdb import set_trace as bp

def MetaDataModule(domain_dim=1, **kwargs):
    if domain_dim == 1:
        return DynamicsDataModule(**kwargs)
    elif domain_dim == 2:
        return Spatial2dDataModule(**kwargs)


def load_dyn_sys_class(dataset_name):
    dataset_classes = {
        'Lorenz63': Lorenz63,
        'Rossler': Rossler,
        'Sinusoid': Sinusoid,
        'ControlledODE': ControlledODE,
        # Add more dataset classes here for other systems

        # add filenames to load
        'darcy_low_res': '../../data/lognormal_N1024_s61.mat',
        'darcy_high_res': '../../data/lognormal_N6024_s421.mat',
    }

    if dataset_name in dataset_classes:
        return dataset_classes[dataset_name]
    else:
        raise ValueError(f"Dataset class '{dataset_name}' not found.")

class DynSys(object):
    def __init__(self, state_dim=1):
        self.state_dim = state_dim
    
    def rhs(self, t, x):
        raise NotImplementedError
    
    def get_inits(self, size):
        raise NotImplementedError

    def solve(self, N_traj, T, dt):
        '''ode solver for the dynamical system'''
        times = torch.arange(0, T, dt)
        xyz0 = self.get_inits(N_traj)
        xyz = odeint(self.rhs, xyz0, times)
        # Size, Seq_len, batch_size, input_dim
        return xyz, times

class Sinusoid(DynSys):
    def __init__(self, freq_low=1, freq_high=1e1, phase=0, state_dim=10):
        super().__init__(state_dim=state_dim)
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.phase = phase

    def solve(self, N_traj, T, dt):
        '''explicit solution for the dynamical system using random frequencies'''
        times = torch.arange(0, T, dt)
        freqs = torch.empty(N_traj, self.state_dim).uniform_(self.freq_low, self.freq_high)
        phases = torch.zeros(N_traj, self.state_dim)

        # evaluate sin(freq * t + phase) for each freq, phase
        freqs = freqs.reshape(freqs.shape[0], freqs.shape[1], 1)
        phases = phases.reshape(*freqs.shape)
        times = times.reshape(1, 1, times.shape[0])

        xyz = torch.sin(2*torch.pi * freqs * times + phases).permute(2, 0, 1)

        # Seq_len, Size (N_traj), state_dim
        return xyz, times.squeeze()

class ControlledODE(DynSys):
    '''dxdt = sin(x) * dudt(t), u(t) = sin(freq * t)'''
    def __init__(self, state_dim=3, freq_low=0.1, freq_high=2):
        super().__init__(state_dim=state_dim)
        self.freq_low = torch.tensor(freq_low)
        self.freq_high = torch.tensor(freq_high)

    def rhs(self, t, x):
        '''
        udot = freq * cos(freq * t),         u(0) = 0
        uddot = - freq**2 * sin(freq * t),   udot(0) = freq
        vdot = sin(v) * udot(t),             v(0) = 0
        '''
        # same as xdot = sin(x) * udot(t); with u(t) = sin(freq * t)
        # x, y = v[:, 0:1], v[:, 1:2]

        u, udot, v = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        du = self.freq * torch.cos(self.freq * t)
        ddu = - self.freq**2 * torch.sin(self.freq * t)
        dv = torch.sin(v) * udot
        return torch.cat([du, ddu, dv], dim=1)

    def get_inits(self, size):

        # sample freq from uniform distribution
        self.freq = torch.empty(size, 1).uniform_(self.freq_low, self.freq_high)

        # sample initial conditions for u, udot, v
        u0 = torch.zeros(size, 1)
        udot0 = self.freq
        v0 = torch.ones(size, 1)

        # concatenate initial conditions
        xyz0 = torch.cat([u0, udot0, v0], dim=1)
        return xyz0

class Lorenz63(DynSys):
    def __init__(self, state_dim=3, sigma=10, rho=28, beta=8/3):
        super().__init__(state_dim=state_dim)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def rhs(self, t, x):
        x, y, z = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return torch.cat([dx, dy, dz], dim=1)
    
    def get_inits(self, size):
        x0 = torch.empty(size, 1).uniform_(-15, 15)
        y0 = torch.empty(size, 1).uniform_(-15, 15)
        z0 = torch.empty(size, 1).uniform_(0, 40)
        xyz0 = torch.cat([x0, y0, z0], dim=1)
        return xyz0

class Rossler(DynSys):
    def __init__(self, state_dim=2, a=0.2, b=0.2, c=5.7):
        super().__init__(state_dim=state_dim)
        self.a = a
        self.b = b
        self.c = c
    
    def rhs(self, t, x):
        x, y, z = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return torch.cat([dx, dy, dz], dim=1)
    
    def get_inits(self, size):
        x0 = torch.empty(size, 1).uniform_(-10, 15)
        y0 = torch.empty(size, 1).uniform_(-30, 0)
        z0 = torch.empty(size, 1).uniform_(0, 30)
        xz0 = torch.cat([x0, y0, z0], dim=1)
        return xz0

class DynamicsDataset(Dataset):
    def __init__(self, size=1000, T=1, sample_rate=0.01, params={},
                 dyn_sys_name='Lorenz63',
                 input_inds=[1], output_inds=[-1],
                 **kwargs):
        '''use params to pass in parameters for the dynamical system'''
        self.size = size
        self.T = T
        self.sample_rate = sample_rate
        self.dynsys = load_dyn_sys_class(dyn_sys_name)(**params)
        self.input_inds = input_inds
        self.output_inds = output_inds

        self.generate_data()
    
    def generate_data(self):
        # Seq_len, Size (N_traj), state_dim
        xyz, times = self.dynsys.solve(N_traj=self.size, T=self.T, dt=self.sample_rate)

        # use traj from the 1st component of L63 as input
        self.x = xyz[:, :, self.input_inds].permute(1, 0, 2)
        # use traj from the 3rd component of L63 as output
        self.y = xyz[:, :, self.output_inds].permute(1, 0, 2)
        # self.x, self.y are both: (n_traj (size), Seq_len, dim_state)

        #normalize data
        self.x_normalizer = UnitGaussianNormalizer(
            self.x.reshape(-1, self.x.shape[-1]).data.numpy())
        self.y_normalizer = UnitGaussianNormalizer(
            self.y.reshape(-1, self.y.shape[-1]).data.numpy())
        self.x = self.x_normalizer.encode(self.x)
        self.y = self.y_normalizer.encode(self.y)

        # currently, times are the same for input and output trajectories
        # and the same across all examples
        self.times_x = times.unsqueeze(-1)
        self.times_y = times.unsqueeze(-1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.times_x, self.times_y

class DynamicsDataModule(pl.LightningDataModule):
    def __init__(self,
            batch_size=64,
            size={'train': 10000, 'val': 500, 'test': 500},
            T={'train': 1, 'val': 1, 'test': 1},
            train_sample_rate=0.01,
            test_sample_rates=[0.01],
            params={},
            dyn_sys_name='Lorenz63',
            input_inds=[0], output_inds=[-1],
            **kwargs
            ):
        super().__init__()
        self.batch_size = batch_size
        self.size = size
        self.T = T
        self.train_sample_rate = train_sample_rate
        self.test_sample_rates = test_sample_rates
        self.params = params
        self.dyn_sys_name = dyn_sys_name
        self.input_inds = input_inds
        self.output_inds = output_inds


    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        self.train = DynamicsDataset(size=self.size['train'],
                                        T=self.T['train'],
                                        sample_rate=self.train_sample_rate,
                                        params=self.params,
                                        dyn_sys_name=self.dyn_sys_name,
                                        input_inds=self.input_inds,
                                        output_inds=self.output_inds)

        self.val = DynamicsDataset(size=self.size['val'],
                                        T=self.T['val'],
                                        sample_rate=self.train_sample_rate,
                                        params=self.params,
                                        dyn_sys_name=self.dyn_sys_name,
                                        input_inds=self.input_inds,
                                        output_inds=self.output_inds)

        # build a dictionary of test datasets with different sample rates
        self.test = {}
        for dt in self.test_sample_rates:
            self.test[dt] = DynamicsDataset(size=self.size['test'],
                                        T=self.T['test'],
                                        sample_rate=dt,
                                        params=self.params,
                                        dyn_sys_name=self.dyn_sys_name,
                                        input_inds=self.input_inds,
                                        output_inds=self.output_inds)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self, sample_rate=None):
        return {dt: DataLoader(self.test[dt], batch_size=self.batch_size) for dt in self.test_sample_rates}

############ Spatial 2D data set ################
class Spatial2dDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=64,
                 split_frac={'train': 0.6, 'val': 0.2, 'test': 0.2},
                 train_sample_rate=2, # strides in this case
                 test_sample_rates=[1,2,4], # strides in this case
                 dyn_sys_name='darcy_low_res',
                 **kwargs
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.train_sample_stride = train_sample_rate
        self.test_sample_rates = test_sample_rates
        self.dyn_sys_name = dyn_sys_name

        self.make_splits(split_frac)

    def make_splits(self, split_frac):
        fname = load_dyn_sys_class(self.dyn_sys_name)
        data = scipy.io.loadmat(fname)
        x = data['input']
        y = data['output']

        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x, y,
            test_size=split_frac['test'])

        # split train_val into train, val
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val,
            train_size=split_frac['train'])

        # define sets
        self.x_train, self.x_val, self.x_test = x_train, x_val, x_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        self.train = Spatial2dDataset(self.x_train, self.y_train,
                            stride=self.train_sample_stride,
                            )

        self.val = Spatial2dDataset(self.x_val, self.y_val,
                            stride=self.train_sample_stride,
                            x_normalizer=self.train.x_normalizer,
                            y_normalizer=self.train.y_normalizer,
                            )

        # build a dictionary of test datasets with different sample rates
        self.test = {}
        for stride in self.test_sample_rates:
            self.test[stride] = Spatial2dDataset(self.x_test, self.y_test,
                                    stride=stride,
                                    x_normalizer=self.train.x_normalizer,
                                    y_normalizer=self.train.y_normalizer,
                                    )
            # NOTE: there is slight train/test leakage because the normalizer
            # sees all the high-frequency training data, and this normalization
            # is used during testing. Technically, we should only store statistics
            # from the explicitly sampled training set.
            # So, not really a "train/test" leakage but rather we should admit that
            # the training technically sees a tiny bit of the high-frequency data
            # (only in the form of its statistics at each coordinate)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self, sample_rate=None):
        return {dt: DataLoader(self.test[dt], batch_size=self.batch_size) for dt in self.test_sample_rates}


class Spatial2dDataset(Dataset):
    def __init__(self, x, y,
                stride=2,
                x_normalizer=None,
                y_normalizer=None,
                **kwargs):
        '''x: (N, length, width)
           y: (N, length, width)
        '''
        self.generate_data(x, y, x_normalizer, y_normalizer, stride)

    def generate_data(self, x, y, x_normalizer, y_normalizer, stride):
        '''x: (N, length, width)
           y: (N, length, width)'''

        # subsample x, y to take all elements on the boundary, and elements in the interior according to stride
        # flattens x, y to be (N, length*width, 1)
        self.active_coordinates_x, x = subsample_and_flatten(x, stride)
        self.active_coordinates_y, y = subsample_and_flatten(y, stride)

        # add extra dimension to x, y so that
        # x: (N, length*width, 1)
        # y: (N, length*width, 1)
        x = x[..., None]
        y = y[..., None]

        # compute normalization
        if x_normalizer is None or y_normalizer is None:
            #normalize data
            self.x_normalizer = UnitGaussianNormalizer(
                x.reshape(-1, x.shape[-1]))
            self.y_normalizer = UnitGaussianNormalizer(
                y.reshape(-1, y.shape[-1]))
        else:
            self.x_normalizer = x_normalizer
            self.y_normalizer = y_normalizer

        # apply normalization
        self.x = self.x_normalizer.encode(x)
        self.y = self.y_normalizer.encode(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.active_coordinates_x, self.active_coordinates_y

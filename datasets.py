import torch
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
import pytorch_lightning as pl

from utils import InactiveNormalizer, UnitGaussianNormalizer, MaxMinNormalizer

def load_dyn_sys_class(dataset_name):
    dataset_classes = {
        'Lorenz63': Lorenz63,
        'Rossler': Rossler,
        'Sinusoid': Sinusoid,
        # Add more dataset classes here for other systems
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
        t = torch.arange(0, T, dt)
        xyz0 = self.get_inits(N_traj)
        xyz = odeint(self.rhs, xyz0, t)
        # Size, Seq_len, batch_size, input_dim
        return xyz

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
        return xyz

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
        xyz = self.dynsys.solve(N_traj=self.size, T=self.T, dt=self.sample_rate)

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

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

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

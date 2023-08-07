import torch
from torch.utils.data import Dataset
from torchdiffeq import odeint

from utils import InactiveNormalizer, UnitGaussianNormalizer, MaxMinNormalizer


class Lorenz63Dataset(Dataset):
    def __init__(self, size=1000, length=100, dt=0.01, sigma=10, rho=28, beta=8/3):
        self.size = size
        self.length = length
        self.dt = dt
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        self.generate_data()

    def generate_data(self):
        # Size, Seq_len, batch_size, input_dim

        x0 = torch.empty(self.size, 1).uniform_(-15, 15)
        y0 = torch.empty(self.size, 1).uniform_(-15, 15)
        z0 = torch.empty(self.size, 1).uniform_(0, 40)
        xyz0 = torch.cat([x0, y0, z0], dim=1)

        t = torch.arange(0, self.length * self.dt, self.dt)

        def lorenz_system(t, xyz):
            x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z
            return torch.cat([dx, dy, dz], dim=1)

        xyz = odeint(lorenz_system, xyz0, t)
        # use traj from the 1st component of L63 as input
        self.x = xyz[:, :, 0:1].permute(1, 0, 2)
        # use traj from the 2nd component of L63 as output (should be easier than 3rd comp)
        self.y = xyz[:, :, 1:2].permute(1, 0, 2)
        # self.x, self.y are both: (n_traj (size), Seq_len, dim_state)

        #normalize data
        self.x_normalizer = UnitGaussianNormalizer(self.x.reshape(-1, self.x.shape[-1]).data.numpy())
        self.y_normalizer = UnitGaussianNormalizer(self.y.reshape(-1, self.y.shape[-1]).data.numpy())
        self.x = self.x_normalizer.encode(self.x)
        self.y = self.y_normalizer.encode(self.y)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
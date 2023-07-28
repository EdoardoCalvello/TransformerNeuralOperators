# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import matplotlib.pyplot as plt
import wandb

from datasets import DynamicsDataset
from models import TransformerEncoder

# Set data hyperparameters
data_hyperparams = {'input_dim': 1,
                    'output_dim': 1,
                    'n_trajectories': {'train': 10000, 'val': 200, 'test': 200},
                    'seq_len': 100,
                    'sample_rate': 0.01,
                    'batch_size': 1000,
                    'dyn_sys_name': 'Rossler',
                    # 'dyn_sys_params': {'sigma': 10, 'rho': 28, 'beta': 8/3},
                    'input_dim': 1,
                    'output_dim': 3,
                    }

# set model hyperparameters
model_hyperparams = {'input_dim': 1,
                        'output_dim': 1,
                        'monitor_metric': 'val_loss',
                        'lr_scheduler_params': {'patience': 2,
                                                'factor': 0.1},
                        'use_transformer': True,
                        'use_positional_encoding': True,
                        'd_model': 128,
                        'nhead': 8,
                        'num_layers': 6,
                        'learning_rate': 0.001,
                        'dropout': 0.01,
                        'dim_feedforward': 128,
                        'activation': 'gelu',
                        }

# set trainer hyperparameters
trainer_hyperparams = {'max_epochs': 100,
                       'log_every_n_steps': 10,
                       'gradient_clip_val': 10.0,
                       'gradient_clip_algorithm': "value",
                       'overfit_batches': 0.0, # overfit to this % of data (default is 0.0 for normal training)...or this many "batches"
                       }

# combine run settings that will be logged to wandb.
list_of_dicts = [data_hyperparams, model_hyperparams, trainer_hyperparams]
all_param_dict = {k: v for d in list_of_dicts for k, v in d.items()}


# Initialize WandB logger
wandb.init(project="lorenz-63-training-transformer-I-v2_x-Predicts-z", config=all_param_dict)
wandb_logger = WandbLogger()

# Load the datasets
train_dataset = DynamicsDataset(size=data_hyperparams['n_trajectories']['train'],
                                **data_hyperparams)
val_dataset = DynamicsDataset(size=data_hyperparams['n_trajectories']['val'], 
                              **data_hyperparams)
test_dataset = DynamicsDataset(size=data_hyperparams['n_trajectories']['test'], 
                               **data_hyperparams)

# Create PyTorch dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=data_hyperparams['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=data_hyperparams['batch_size'])
test_loader = DataLoader(test_dataset, batch_size=data_hyperparams['batch_size'])

# Initialize the model
model = TransformerEncoder(**model_hyperparams)

# Create a PyTorch Lightning trainer with the WandbLogger
# used this link to find LRmonitor: https://community.wandb.ai/t/how-to-log-the-learning-rate-with-pytorch-lightning-when-using-a-scheduler/3964/5
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(logger=wandb_logger, callbacks=[lr_monitor], **trainer_hyperparams)

# Train the model
trainer.fit(model, train_loader, val_loader)

# Test the model
trainer.test(model, test_loader)

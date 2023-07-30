# Import deep learning modules
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb

# Import custom modules
from datasets import DynamicsDataset
from models import TransformerEncoder

class Runner:
    def __init__(self,
            project_name="Rossler_x-Predicts-z",
            input_dim_data=1,
            output_dim_data=1,
            n_trajectories_train=10000,
            n_trajectories_val=200,
            n_trajectories_test=200,
            seq_len=100,
            sample_rate=0.01,
            batch_size=1000,
            dyn_sys_name='Rossler',
            input_dim_model=1,
            output_dim_model=1,
            monitor_metric='val_loss',
            lr_scheduler_params={'patience': 2, 'factor': 0.1},
            use_transformer=True,
            use_positional_encoding=True,
            d_model=128,
            nhead=8,
            num_layers=6,
            learning_rate=0.001,
            dropout=0.01,
            dim_feedforward=128,
            activation='gelu',
            max_epochs=100,
            log_every_n_steps=10,
            gradient_clip_val=10.0,
            gradient_clip_algorithm="value",
            overfit_batches=0.0):

        self.project_name = project_name

        self.data_hyperparams = {'n_trajectories': {'train': n_trajectories_train,
                                                    'val': n_trajectories_val,
                                                    'test': n_trajectories_test,
                                                    },
                                 'seq_len': seq_len,
                                 'sample_rate': sample_rate,
                                 'batch_size': batch_size,
                                 'dyn_sys_name': dyn_sys_name,
                                 'input_dim': input_dim_data,
                                 'output_dim': output_dim_data,
                                 }
        
        self.model_hyperparams = {'input_dim': input_dim_model,
                                  'output_dim': output_dim_model,
                                  'monitor_metric': monitor_metric,
                                  'lr_scheduler_params': lr_scheduler_params,
                                  'use_transformer': use_transformer,
                                  'use_positional_encoding': use_positional_encoding,
                                  'd_model': d_model,
                                  'nhead': nhead,
                                  'num_layers': num_layers,
                                  'learning_rate': learning_rate,
                                  'dropout': dropout,
                                  'dim_feedforward': dim_feedforward,
                                  'activation': activation,
                                  }
        
        self.trainer_hyperparams = {'max_epochs': max_epochs,
                                    'log_every_n_steps': log_every_n_steps,
                                    'gradient_clip_val': gradient_clip_val,
                                    'gradient_clip_algorithm': gradient_clip_algorithm,
                                    'overfit_batches': overfit_batches,
                                    }
        
        self.run()

    def run(self):
        # Combine run settings that will be logged to wandb.
        list_of_dicts = [self.data_hyperparams,
                         self.model_hyperparams, self.trainer_hyperparams]
        all_param_dict = {k: v for d in list_of_dicts for k, v in d.items()}

        # Initialize WandB logger
        wandb.init(
            project=self.project_name, config=all_param_dict)
        wandb_logger = WandbLogger()

        # Load the datasets
        train_dataset = DynamicsDataset(size=self.data_hyperparams['n_trajectories']['train'],
                                        **self.data_hyperparams)
        val_dataset = DynamicsDataset(size=self.data_hyperparams['n_trajectories']['val'],
                                      **self.data_hyperparams)
        test_dataset = DynamicsDataset(size=self.data_hyperparams['n_trajectories']['test'],
                                       **self.data_hyperparams)

        # Create PyTorch dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.data_hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=self.data_hyperparams['batch_size'])
        test_loader = DataLoader(
            test_dataset, batch_size=self.data_hyperparams['batch_size'])

        # Initialize the model
        model = TransformerEncoder(**self.model_hyperparams)

        # Create a PyTorch Lightning trainer with the WandbLogger
        # used this link to find LRmonitor: https://community.wandb.ai/t/how-to-log-the-learning-rate-with-pytorch-lightning-when-using-a-scheduler/3964/5
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(logger=wandb_logger, callbacks=[
                            lr_monitor], **self.trainer_hyperparams)

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # Test the model
        trainer.test(model, test_loader)
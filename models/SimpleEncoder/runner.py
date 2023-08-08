# Import deep learning modules
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import wandb
from pytorch_lightning.tuner import Tuner

# Import custom modules
from datasets import DynamicsDataModule
from models.SimpleEncoder.SimpleEncoder_lightning import SimpleEncoderModule

class Runner:
    def __init__(self,
            seed=0,
            deterministic=True, # set to False to get different results each time
            project_name="Rossler_x-Predicts-z",
            input_inds=[0],
            output_inds=[-1],
            n_trajectories_train=10000,
            n_trajectories_val=200,
            n_trajectories_test=200,
            seq_len=100,
            sample_rate=0.01,
            batch_size=32,
            tune_batch_size=True,
            dyn_sys_name='Rossler',
            monitor_metric='val_loss',
            lr_scheduler_params={'patience': 2, 'factor': 0.1},
            tune_initial_lr=True,
            use_transformer=True,
            use_positional_encoding=True,
            include_y0_input=False,
            do_layer_norm=True,
            norm_first=False,
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

        seed_everything(seed, workers=True)

        self.project_name = project_name

        self.data_hyperparams = {'size': {'train': n_trajectories_train,
                                                    'val': n_trajectories_val,
                                                    'test': n_trajectories_test,
                                                    },
                                 'seq_len': {'train': seq_len,
                                             'val': seq_len,
                                             'test': seq_len,},
                                 'sample_rate': {'train': sample_rate,
                                                 'val': sample_rate,
                                                 'test': sample_rate,},
                                 'batch_size': batch_size,
                                 'tune_batch_size': tune_batch_size,
                                 'dyn_sys_name': dyn_sys_name,
                                 'input_inds': input_inds,
                                 'output_inds': output_inds,
                                 }

        self.model_hyperparams = {'input_dim': len(input_inds),
                                  'output_dim': len(output_inds),
                                  'monitor_metric': monitor_metric,
                                  'lr_scheduler_params': lr_scheduler_params,
                                  'use_transformer': use_transformer,
                                  'use_positional_encoding': use_positional_encoding,
                                  'include_y0_input': include_y0_input,
                                  'do_layer_norm': do_layer_norm,
                                  'norm_first': norm_first,
                                  'd_model': d_model,
                                  'nhead': nhead,
                                  'num_layers': num_layers,
                                  'learning_rate': learning_rate,
                                  'dropout': dropout,
                                  'dim_feedforward': dim_feedforward,
                                  'activation': activation,
                                  # add extra sequence to allow for inclusion of output I.C.
                                  'max_sequence_length': seq_len + len(output_inds),
                                  }
        
        self.trainer_hyperparams = {'max_epochs': max_epochs,
                                    'log_every_n_steps': log_every_n_steps,
                                    'gradient_clip_val': gradient_clip_val,
                                    'gradient_clip_algorithm': gradient_clip_algorithm,
                                    'overfit_batches': overfit_batches,
                                    'deterministic': deterministic,
                                    }
        
        self.other_hyperparams = {'seed': seed, 'tune_initial_lr': tune_initial_lr,
                                  }

        self.run()

    def run(self):
        # Combine run settings that will be logged to wandb.
        list_of_dicts = [self.other_hyperparams,
                         self.data_hyperparams,
                         self.model_hyperparams,
                         self.trainer_hyperparams]
        all_param_dict = {k: v for d in list_of_dicts for k, v in d.items()}

        # Initialize WandB logger
        wandb.init(
            project=self.project_name, config=all_param_dict)
        wandb_logger = WandbLogger()

        # Load the DataModule
        datamodule = DynamicsDataModule(**self.data_hyperparams)

        # Initialize the model
        model = SimpleEncoderModule(**self.model_hyperparams)

        # Set callbacks for trainer (lr monitor, early stopping)

        # Create a PyTorch Lightning trainer with the WandbLogger
        # used this link to find LRmonitor: https://community.wandb.ai/t/how-to-log-the-learning-rate-with-pytorch-lightning-when-using-a-scheduler/3964/5
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # Create an early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=True)

        # aggregate all callbacks
        callbacks = [lr_monitor, early_stopping]


        # Initialize the trainer
        trainer = Trainer(logger=wandb_logger, callbacks=callbacks,
                              **self.trainer_hyperparams)

        # Tune the model
        tuner = Tuner(trainer)

        # Tune the batch size
        # half the identified batch size to avoid maxing out RAM
        if self.data_hyperparams['tune_batch_size']:
            if torch.cuda.is_available():
                print('Using GPU, so setting batch size scaler max_trials to 25 (pick a smaller number if this destroys the machine)')
                max_trials = 25
            else:
                print('Using CPU, so setting batch size scaler max_trials to 6 (avoid maxing RAM on a local machine)')
                max_trials = 6
            tuner.scale_batch_size(model, max_trials=max_trials, datamodule=datamodule)
            datamodule.batch_size = max(1, datamodule.batch_size // 2)
            print('Using batch size: ', datamodule.batch_size)

        # Tune the learning rate
        if self.other_hyperparams['tune_initial_lr']:
            min_lr, max_lr = model.learning_rate * 0.01, model.learning_rate * 100
            tuner.lr_find(model, datamodule=datamodule, min_lr=min_lr, max_lr=max_lr, num_training=20)
            print('Using learning rate: ', model.learning_rate)

        # Train the model
        trainer.fit(model, datamodule=datamodule)

        # Test the model
        trainer.test(model, datamodule=datamodule)

from runner import Runner
from utils import dict_combiner
import argparse

# use argparse to get command line argument for which experiment to run
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default='transformer-sweep1')
parser.add_argument('--n', type=int, default=0)
args = parser.parse_args()

# build a dict of experimental conditions
exp_dict = {
    'project_name': [args.project_name],
    # data settings
    'n_trajectories_train': [1000, 10000],
    'n_trajectories_val': [500],
    'n_trajectories_test': [500],
    'seq_len': [100],
    'sample_rate': [0.01],
    'batch_size': [100, 1000, 10000],
    'dyn_sys_name': ['Rossler','Lorenz63'],
    'input_dim_data': [1],
    'output_dim_data': [3],
    # optimizer settings
    'learning_rate': [1e-3, 1e-4],
    'dropout': [1e-2, 1e-4],
    'lr_scheduler_params': [{'patience': 2, 'factor': 0.1},
                            {'patience': 2, 'factor': 0.5}
                             ],
    'max_epochs': [100],
    'monitor_metric': ['val_loss'],
    # model settings
    'd_model': [128, 1024],
    'nhead': [2, 8],
    'num_layers': [2, 6],
    'dim_feedforward': [128, 1024],
    'activation': ['gelu'],
}

exp_list = dict_combiner(exp_dict)

# run the experiment
Runner(**exp_list[args.n])




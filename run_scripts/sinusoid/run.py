import sys
sys.path.append('../../')
from models.SimpleEncoder.runner import Runner
from utils import dict_combiner
import argparse

# use argparse to get command line argument for which experiment to run
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default='sinusoid')
parser.add_argument('--id', type=int, default=0)
args = parser.parse_args()

# build a dict of experimental conditions
exp_dict = {
    'project_name': [args.project_name],
    # data settings
    'n_trajectories_train': [5000], # smaller dataset for debugging
    'n_trajectories_val': [500],
    'n_trajectories_test': [500],
    'T': [1],
    'train_sample_rate': [0.01],
    'test_sample_rates': [[0.5e-2, 1e-2, 2e-2]],
    'batch_size': [64],
    'dyn_sys_name': ['Sinusoid'],
    'input_inds': [[0]],
    'output_inds': [[0],[1]],
    # optimizer settings
    'learning_rate': [1e-3],
    'dropout': [1e-4],
    'lr_scheduler_params': [
                            {'patience': 2, 'factor': 0.5},
                             ],
    'max_epochs': [100],
    'monitor_metric': ['loss/val/mse'],
    # model settings (modest model size for debugging)
    'd_model': [128],
    'nhead': [8],
    'num_layers': [6],
    'dim_feedforward': [128],
    'activation': ['gelu'],
}

exp_list = dict_combiner(exp_dict)

# Print the length of the experiment list
print('Number of experiments to sweep: ', len(exp_list))

# run the experiment
Runner(**exp_list[args.id])




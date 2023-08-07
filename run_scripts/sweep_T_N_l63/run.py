import sys
sys.path.append('../../')
from models.SimpleEncoder.runner import Runner
from utils import dict_combiner
import argparse

# use argparse to get command line argument for which experiment to run
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str,
                    default='sweep_sequence_length-fixed_data_quantity')
parser.add_argument('--id', type=int, default=0)
args = parser.parse_args()


T_DATA = 200*1000

# build a dict of experimental conditions
exp_dict = {
    'project_name': [args.project_name],
    # data settings
    'n_trajectories_train': [1000], # smaller dataset for debugging
    'n_trajectories_val': [100],
    'n_trajectories_test': [100],
    'seq_len': [5, 10, 50, 100, 200, 500, 1000],
    'sample_rate': [0.01],
    'batch_size': [128],
    'dyn_sys_name': ['Lorenz63'],
    'input_inds': [[0]],
    'output_inds': [[1,2]],
    # optimizer settings
    'learning_rate': [1e-3],
    'dropout': [1e-2],
    'lr_scheduler_params': [
                            {'patience': 2, 'factor': 0.5},
                             ],
    'max_epochs': [70],
    'monitor_metric': ['val_loss'],
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

# extract the experiment of interest
exp = exp_list[args.id]

# set the number of trajectories to use for training based on the sequence length
exp['n_trajectories_train'] = int(T_DATA/exp['seq_len'])

# run the experiment
Runner(**exp)




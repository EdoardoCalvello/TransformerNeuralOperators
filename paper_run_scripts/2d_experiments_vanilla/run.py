import sys
sys.path.append('../../')
from models.SimpleEncoder.runner import Runner
from utils import dict_combiner
import argparse

# use argparse to get command line argument for which experiment to run
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default='2d_paper_experiments_Vanilla')
parser.add_argument('--id', type=int, default=0)
args = parser.parse_args()

# once working, need to update readme with:
# conda install -c anaconda scikit-learn

# build a dict of experimental conditions
exp_dict = {
    'project_name': [args.project_name],
    # data settings
    'split_frac': [{'train': 0.9, 'val': 0.05, 'test': 0.05}],
    'random_state': [0],
    'domain_dim': [2], # 1 for timeseries, 2 for 2D spatial
    'train_sample_rate': [1],
    'test_sample_rates': [[0.5,1,2]],
    'pos_enc_coeff': [1],#[0, 1, 2, 3],
    'batch_size': [1,4,8,16],
    'dyn_sys_name': ['darcy_low_res'],
    # optimizer settings
    'learning_rate': [1e-4],
    'dropout': [1e-4],
    'lr_scheduler_params': [
                            {'patience': 2, 'factor': 0.5},
                             ],
    'max_epochs': [100],
    'monitor_metric': ['loss/val/mse'],
    # model settings (modest model size for debugging)
    'include_y0_input': [False], #['uniform', 'staggered', False],
    'd_model': [128],
    'nhead': [8],
    'num_layers': [6],
    'dim_feedforward': [128],
    'activation': ['gelu'],
    'use_positional_encoding': [False], #['continuous', 'discrete', False],
    'append_position_to_x': [True], #[True, False],
}

exp_list = dict_combiner(exp_dict)

# Print the length of the experiment list
print('Number of experiments to sweep: ', len(exp_list))

# run the experiment
Runner(**exp_list[args.id])

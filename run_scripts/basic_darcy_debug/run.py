import sys
sys.path.append('../../')
from models.SimpleEncoder.runner import Runner
from utils import dict_combiner
import argparse

# use argparse to get command line argument for which experiment to run
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default='darcy_flow_1.0')
parser.add_argument('--id', type=int, default=0)
args = parser.parse_args()

# once working, need to update readme with:
# conda install -c anaconda scikit-learn

# build a dict of experimental conditions
exp_dict = {
    'project_name': [args.project_name],
    # data settings
    'split_frac': [{'train': 0.6, 'val': 0.2, 'test': 0.2}],
    'domain_dim': [2], # 1 for timeseries, 2 for 2D spatial
    'train_sample_rate': [2],
    'test_sample_rates': [[1,2,4]],
    'batch_size': [4, 16, 64],
    'dyn_sys_name': ['darcy_low_res'],
    # optimizer settings
    'learning_rate': [1e-3],
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
}

exp_list = dict_combiner(exp_dict)

# Print the length of the experiment list
print('Number of experiments to sweep: ', len(exp_list))

# run the experiment
Runner(**exp_list[args.id])




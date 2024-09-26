import sys
sys.path.append('../../')
from models.FANO.runner import Runner
from utils import dict_combiner
import argparse

# use argparse to get command line argument for which experiment to run
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default='2d_paper_experiments_FANO')
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
    'test_sample_rates': [[0.5,0.75,1,1.5,2]],
    'test_im_sizes': [[832,624,416,312,208]],
    'test_patch_sizes': [[64,48,32,24,16]],
    'batch_size': [2],
    'dyn_sys_name': ['NavierStokes'],
    # optimizer settings
    'learning_rate': [1e-3,1e-4],
    'dropout': [1e-4],
    'lr_scheduler_params': [
                            {'patience': 2, 'factor': 0.5},
                             ],
    'max_epochs': [80],
    'monitor_metric': ['loss/val/mse'],
    # model settings (modest model size for debugging)
    'patch': [True],
    'patch_size': [32],
    'modes': [[9,9]],
    'im_size': [416],
    'd_model': [32,64,96],
    'nhead': [8],
    'num_layers': [6],
    'dim_feedforward': [128],
    'activation': ['gelu'],
    'gradient_clip_val':[None] #or 10.0 whenever want to use
}

exp_list = dict_combiner(exp_dict)

# Print the length of the experiment list
print('Number of experiments to sweep: ', len(exp_list))

# run the experiment
Runner(**exp_list[args.id])
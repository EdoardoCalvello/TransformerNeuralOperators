import sys
sys.path.append('../../')

import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

from models.FANO.FANO_lightning import SimpleEncoderModule
from datasets import MetaDataModule

import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_path = '../../paper_run_scripts/2d_experiments_FANO/lightning_logs/4kahzdf3/checkpoints/epoch=79-step=288000.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=device)

def kernel_ft(x, coords_x):

    N, rows, cols = x.shape
    coords = coords_x[0,...].permute(0,2,1)*(rows-1)
    coords = torch.sum(coords**2, dim=0)[:, :cols//2+1]

    return coords


def forward_smoothing(x, coords_x):

    x_ft = torch.fft.rfft2(x)
    scale, decay = 0.05, 1
    x_ft = x_ft * (1/((1+(scale**2)*(4*(np.pi**2)/(x.shape[1]**2))*kernel_ft(x,coords_x))**decay))
    x = torch.fft.irfft2(x_ft, s=(x.size(-2), x.size(-1)))

    return x


api = wandb.Api()
run = api.run('edoardo-calvello/2d_paper_experiments_FANO/4kahzdf3')

# Load the model parameters
config = run.config
# Create a dictionary of valid arguments for the model
valid_model_args = {key: value for key, value in config.items() if key in SimpleEncoderModule.__init__.__code__.co_varnames}
# Create the model using the same architecture
model = SimpleEncoderModule(**valid_model_args)
model.first_forward = False
model = model.to(device)
# Load the model weights from Wandb
model.load_state_dict(checkpoint['state_dict'])

model.model.set_im_size(208,16)

# Load the data module
datamodule = MetaDataModule(**config)
datamodule.setup(stage='test')  # Call the setup method to set up the test data
test_data_loader = datamodule.test_dataloader()  # Retrieve the test data loader
#change the [1] depending on the sample rates
test_sample_rate = 2
test_data_loader = test_data_loader[test_sample_rate]

# Set the model to evaluation mode
model.eval()


errors = []
predictions = []
# Iterate through the test data and make predictions, computing errors

for sample in test_data_loader.dataset:
    x, y, coords_x, coords_y, x_train_fourier_normalizer = sample
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    coords_x = torch.from_numpy(coords_x).to(device)
    coords_y = torch.from_numpy(coords_y).to(device)
    with torch.no_grad():
        # Forward pass to get predictions
        y_pred = model(x.unsqueeze(0), y=y.unsqueeze(0),
                        coords_x=coords_x.unsqueeze(0), coords_y=coords_y.unsqueeze(0), x_train_fourier_normalizer=None)
        
        #print(y_pred.shape)
        
        #y_pred = forward_smoothing(y_pred, coords_x.unsqueeze(0))

        # Compute relative error
        ############################################
        ############################################
        error_sample = torch.sqrt(torch.mean(torch.abs(y_pred.to(device) - (y.unsqueeze(0)))**2))/torch.sqrt(
            torch.mean(torch.abs(y.unsqueeze(0))**2))

        ############################################
        ############################################
    
        predictions.append(y_pred.to('cpu'))
        errors.append(error_sample.to('cpu'))

# Convert the NumPy array to a PyTorch tensor
predictions = np.array(predictions)
# Convert the list of errors to a NumPy array
errors = np.array(errors)

# Find the indices of the samples with the median and lowest error
median_idx = np.argsort(errors)[len(errors) // 2]
median_error= errors[median_idx]
min_error_idx = np.argmax(errors)
min_error = errors[min_error_idx]

# Retrieve the corresponding data for these samples
x_median, y_median, coords_x_median, coords_y_median, _ = test_data_loader.dataset[median_idx]
x_min, y_min, coords_x_min, coords_y_min, _ = test_data_loader.dataset[min_error_idx]

y_pred_min = predictions[min_error_idx]
y_pred_median = predictions[median_idx]
# Now you have the data for the samples with median and lowest errors (x_median, y_median, x_min_loss, y_min_loss).


############################################################################################################
############################################################################################################

 # Each element of y_true and y_pred is a 2D field with coordinates given by coords_y
 # plot the values of y_true and y_pred at the indices given by coords_y
n_grid=416

coords_x = coords_x.cpu()
coords_y = coords_y.cpu()


# get the low and high indices of the y coordinates
i_low_1, i_low_2 = np.min(coords_y_min[...,0]), np.min(coords_y_min[...,1])
i_high_1, i_high_2 = np.max(coords_y_min[...,0]), np.max(coords_y_min[...,1])
# build a meshgrid of coordinates based on coords_y
y1i_min, y2i_min = np.meshgrid(np.linspace(i_low_1, i_high_1, n_grid), np.linspace(i_low_2,i_high_2, n_grid))

# get the low and high indices of the x coordinates
i_low_1, i_low_2 = np.min(coords_x_min[...,0]), np.min(coords_x_min[...,1])
i_high_1, i_high_2 = np.max(coords_x_min[...,0]), np.max(coords_x_min[...,1])
#build a meshgrid of coordinates based on coords_y, ordering...
x1i_min, x2i_min = np.meshgrid(np.linspace(i_low_1, i_high_1, n_grid), np.linspace(i_low_2,i_high_2, n_grid))

# get the low and high indices of the y coordinates
i_low_1_median, i_low_2_median = np.min(coords_y_median[...,0]), np.min(coords_y_median[...,1])
i_high_1_median, i_high_2_median = np.max(coords_y_median[...,0]), np.max(coords_y_median[...,1])
# build a meshgrid of coordinates based on coords_y
y1i_median, y2i_median = np.meshgrid(np.linspace(i_low_1_median, i_high_1_median, n_grid), np.linspace(i_low_2_median,i_high_2_median, n_grid))

# get the low and high indices of the x coordinates
i_low_1_median, i_low_2_median = np.min(coords_x_median[...,0]), np.min(coords_x_median[...,1])
i_high_1_median, i_high_2_median = np.max(coords_x_median[...,0]), np.max(coords_x_median[...,1])
#build a meshgrid of coordinates based on coords_y, ordering...
x1i_median, x2i_median = np.meshgrid(np.linspace(i_low_1_median, i_high_1_median, n_grid), np.linspace(i_low_2_median,i_high_2_median, n_grid))

plt.figure()
fig, axs = plt.subplots(
    nrows=2, ncols=4, figsize=(24,12), sharex=True, squeeze=False)

x_input_min = griddata(
                (coords_x_min[0, :, :].flatten(), coords_x_min[1, :, :].flatten()), x_min.flatten(), (x1i_min, x2i_min), method='linear')
y_true_min = griddata(
                (coords_y_min[0, :, :].flatten(), coords_y_min[1, :, :].flatten()), y_min.flatten(), (y1i_min, y2i_min), method='linear')
y_pred_min = griddata(
                (coords_y_min[0, :, :].flatten(), coords_y_min[1, :, :].flatten()), y_pred_min.flatten(), (y1i_min, y2i_min), method='linear')

y_rel_diff_min = np.abs(y_pred_min - y_true_min)

x_input_median = griddata(
                (coords_x_median[0, :, :].flatten(), coords_x_median[1, :, :].flatten()), x_median.flatten(), (x1i_median, x2i_median), method='linear')
y_true_median = griddata(
                (coords_y_median[0, :, :].flatten(), coords_y_median[1, :, :].flatten()), y_median.flatten(), (y1i_median, y2i_median), method='linear')
y_pred_median = griddata(
                (coords_y_median[0, :, :].flatten(), coords_y_median[1, :, :].flatten()), y_pred_median.flatten(), (y1i_median, y2i_median), method='linear')

y_rel_diff_median = np.abs(y_pred_median - y_true_median)

y_pred_x_diff_median = np.abs(y_pred_median - x_input_median)
y_x_diff_median = np.abs(y_true_median - x_input_median)
y_pred_x_diff_min = np.abs(y_pred_min - x_input_min)
y_x_diff_min = np.abs(y_true_min - x_input_min)


data_sets = {
            (1, 0): x_input_min,
            (1, 1): y_x_diff_min,
            (1, 2): y_pred_x_diff_min,
            (1, 3): np.log10(y_rel_diff_min),
            (0, 0): x_input_median,
            (0, 1): y_x_diff_median,
            (0, 2): y_pred_x_diff_median,
            (0, 3): np.log10(y_rel_diff_median)
        }

for (i, j), data in data_sets.items():
    bounds = (-7, 1) if j == 3 else (None, None)
    im = axs[i, j].imshow(data, cmap='viridis' if j < 3 else 'inferno', vmin=bounds[0], vmax=bounds[1])
    if j == 3:
        error_info = f"Median Relative L2 Error ({median_error:.2e})" if i == 0 else f"Maximum Relative L2 Error ({min_error:.2e})"
    else:
        error_info = ""
    axs[i, j].set_title(
        f"{'Input Vorticity' if j == 0 else 'Difference Between Truth and Input' if j == 1 else 'Difference Between Prediction and Input' if j == 2 else 'Pointwise Absolute Error'}\n "
        f"{error_info}",
        fontsize=14
    )
    axs[i, j].set_xticks(np.linspace(0, n_grid, num=3))
    axs[i, j].set_xticklabels([f"${i:.0f}\pi$" for i in np.linspace(0, 2, num=3)])
    axs[i, j].set_yticks(np.linspace(0, n_grid, num=3))
    axs[i, j].set_yticklabels([f"${i:.0f}\pi$" for i in np.linspace(0, 2, num=3)])

    fig.colorbar(im, ax=axs[i, j], shrink=0.6)

fig.suptitle('Median and Maximum Relative L2 Error Samples', fontsize=30)
fig.subplots_adjust(top=0.9)
plt.subplots_adjust(hspace=-0.1,wspace=0.1)
plt.savefig(f"test_error_plots_dt{test_sample_rate}.png")

############################################################################################################
############################################################################################################
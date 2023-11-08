import sys
sys.path.append('../')

import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

from models.SimpleEncoder.SimpleEncoder_lightning import SimpleEncoderModule
from datasets import MetaDataModule

import pdb

#plot - pointwise relative errors

project_name = "darcy_flow_1.0"
run_name = "curious-morning-63"
run_id = "ojkji264"
#artifact name = run name _ epoch number .pt:v0
artifact_name = "curious-morning-63_epoch99.pt:v0"
#file name = run name _ epoch number .pt
file_name = 'curious-morning-63_epoch99.pt'

api = wandb.Api()

# Initialize Wandb
run = api.run(f"edoardo-calvello/{project_name}/{run_id}")
# Fetch the specified artifact
#artifact = run.use_artifact(f"edoardo-calvello/{project_name}/{artifact_name}", type="model")
artifact = api.artifact(f"edoardo-calvello/{project_name}/{artifact_name}")
# Download the specified artifact
artifact_dir = artifact.download()
# Retrieve the model file from the downloaded artifact directory
model_file_path = f"{artifact_dir}/{file_name}"
# Load the model parameters
config = run.config
# Create a dictionary of valid arguments for the model
valid_model_args = {key: value for key, value in config.items() if key in SimpleEncoderModule.__init__.__code__.co_varnames}
# Create the model using the same architecture
model = SimpleEncoderModule(**valid_model_args)
# Load the model weights from Wandb
model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))

config['test_sample_rates'] = [1,2,4]

# Load the data module
datamodule = MetaDataModule(**config)
datamodule.setup(stage='test')  # Call the setup method to set up the test data
test_data_loader = datamodule.test_dataloader()  # Retrieve the test data loader
#change the [1] depending on the sample rates
test_data_loader = test_data_loader[4]

# Set the model to evaluation mode
model.eval()


errors = []
predictions = []


# Iterate through the test data and make predictions, computing errors

for sample in test_data_loader.dataset:
    x, y, coords_x, coords_y = sample
    with torch.no_grad():
        # Forward pass to get predictions
        y_pred = model(torch.from_numpy(x).unsqueeze(0), y=torch.from_numpy(y).unsqueeze(0),
                        coords_x=torch.from_numpy(coords_x).unsqueeze(0), coords_y=torch.from_numpy(coords_y).unsqueeze(0))
        # Compute relative error
        ############################################
        ############################################
        error_sample = torch.sqrt(torch.mean(torch.abs(y_pred - (torch.from_numpy(y).unsqueeze(0)))**2))/torch.sqrt(
            torch.mean(torch.abs(torch.from_numpy(y).unsqueeze(0))**2))

        ############################################
        ############################################
    
        predictions.append(y_pred)
        errors.append(error_sample)
        

# Convert the NumPy array to a PyTorch tensor
predictions = np.array(predictions)
# Convert the list of errors to a NumPy array
errors = np.array(errors)

# Find the indices of the samples with the median and lowest error
median_idx = np.argsort(errors)[len(errors) // 2]
print(errors[median_idx])
min_error_idx = np.argmax(errors)
print(errors[min_error_idx])

# Retrieve the corresponding data for these samples
x_median, y_median, coords_x_median, coords_y_median = test_data_loader.dataset[median_idx]
x_min, y_min, coords_x_min, coords_y_min = test_data_loader.dataset[min_error_idx]

y_pred_min = predictions[min_error_idx]
y_pred_median = predictions[median_idx]
# Now you have the data for the samples with median and lowest errors (x_median, y_median, x_min_loss, y_min_loss).


############################################################################################################
############################################################################################################

 # Each element of y_true and y_pred is a 2D field with coordinates given by coords_y
 # plot the values of y_true and y_pred at the indices given by coords_y

n_grid=100

# get the low and high indices of the y coordinates
i_low_1, i_low_2 = np.min(coords_y_min[...,0]), np.min(coords_y_min[...,1])
i_high_1, i_high_2 = np.max(coords_y_min[...,0]), np.max(coords_y_min[...,1])
# build a meshgrid of coordinates based on coords_y
y1i_min, y2i_min = np.meshgrid(np.linspace(i_low_1, i_high_1, n_grid), np.linspace(i_low_2, i_high_2, n_grid))

# get the low and high indices of the x coordinates
i_low_1, i_low_2 = np.min(coords_x_min[...,0]), np.min(coords_x_min[...,1])
i_high_1, i_high_2 = np.max(coords_x_min[...,0]), np.max(coords_x_min[...,1])
# build a meshgrid of coordinates based on coords_y
x1i_min, x2i_min = np.meshgrid(np.linspace(i_low_1, i_high_1, n_grid), np.linspace(i_low_2, i_high_2, n_grid))

# get the low and high indices of the y coordinates
i_low_1_median, i_low_2_median = np.min(coords_y_median[...,0]), np.min(coords_y_median[...,1])
i_high_1_median, i_high_2_median = np.max(coords_y_median[...,0]), np.max(coords_y_median[...,1])
# build a meshgrid of coordinates based on coords_y
y1i_median, y2i_median = np.meshgrid(np.linspace(i_low_1_median, i_high_1_median, n_grid), np.linspace(i_low_2_median, i_high_2_median, n_grid))

# get the low and high indices of the x coordinates
i_low_1_median, i_low_2_median = np.min(coords_x_median[...,0]), np.min(coords_x_median[...,1])
i_high_1_median, i_high_2_median = np.max(coords_x_median[...,0]), np.max(coords_x_median[...,1])
# build a meshgrid of coordinates based on coords_y
x1i_median, x2i_median = np.meshgrid(np.linspace(i_low_1_median, i_high_1_median, n_grid), np.linspace(i_low_2_median, i_high_2_median, n_grid))

plt.figure()
fig, axs = plt.subplots(
    nrows=4, ncols=2, figsize=(16, 24), sharex=True, squeeze=False)

x_input_min = griddata(
        (coords_x_min[:, 0], coords_x_min[:, 1]), x_min, (x1i_min, x2i_min), method='linear')
y_true_min = griddata(
        (coords_y_min[ :, 0], coords_y_min[ :, 1]), y_min, (y1i_min, y2i_min), method='linear')
y_pred_min = griddata((coords_y_min[ :, 0], coords_y_min[ :, 1]), y_pred_min[0,:,:], (y1i_min, y2i_min), method='linear')
y_rel_diff_min = np.abs(y_pred_min - y_true_min)

x_input_median = griddata(
        (coords_x_median[:, 0], coords_x_median[:, 1]), x_median, (x1i_median, x2i_median), method='linear')
y_true_median = griddata(
        (coords_y_median[:, 0], coords_y_median[:, 1]), y_median, (y1i_median, y2i_median), method='linear')
y_pred_median = griddata((coords_y_median[ :, 0], coords_y_median[ :, 1]), y_pred_median[0,:,:], (y1i_median, y2i_median), method='linear')
y_rel_diff_median = np.abs(y_pred_median - y_true_median)


data_sets = {
    (0, 1): x_input_min,
    (1, 1): y_true_min,
    (2, 1): y_pred_min,
    (3, 1): np.log10(y_rel_diff_min),
    (0, 0): x_input_median,
    (1, 0): y_true_median,
    (2, 0): y_pred_median,
    (3, 0): np.log10(y_rel_diff_median)
}

for (i, j), data in data_sets.items():
    bounds = (-7, 1) if i == 3 else (None, None)
    im = axs[i, j].imshow(data, cmap='viridis' if i < 3 else 'inferno', vmin=bounds[0], vmax=bounds[1])
    axs[i, j].set_title(
        f"{'Input Field' if i == 0 else 'Ground Truth' if i == 1 else 'Prediction' if i == 2 else 'Pointwise Error'} "
        f"{'Median' if j == 0 else 'Worst'}"
    )

    cbar = plt.colorbar(im, ax=axs[i, j])

fig.suptitle('Predicted Fields: Prediction vs. Truth')
plt.subplots_adjust(hspace=0.5, wspace=0)
plt.savefig("test_error_plots.png")

############################################################################################################
############################################################################################################


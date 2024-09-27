import numpy as np
from scipy.interpolate import griddata
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt

from models.TNO.TNO_pytorch import SimpleEncoder

# Define the pytorch lightning module for training the Simple Encoder model
class SimpleEncoderModule(pl.LightningModule):
    def __init__(self, input_dim=1, output_dim=1, d_model=32, nhead=8, num_layers=6,
                 domain_dim=1, # 1 for timeseries, 2 for spatial 2D
                 learning_rate=0.01, max_sequence_length=100,
                 do_layer_norm=True,
                 use_transformer=True,
                 use_positional_encoding='continuous',
                 append_position_to_x=False,
                 pos_enc_coeff=2,
                 include_y0_input=False,
                 loss_name = 'l2_rel', # 'l2_rel' or 'l2'
                 activation='relu',
                 monitor_metric='train_loss',
                 lr_scheduler_params={'patience': 3,
                                      'factor': 0.5},
                 dropout=0.1,
                 norm_first=False,
                 dim_feedforward=2048):
        super(SimpleEncoderModule, self).__init__()
        self.first_forward = True # for plotting model-related things once at beginnning of training
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.use_transformer = use_transformer
        self.use_positional_encoding = use_positional_encoding
        self.append_position_to_x = append_position_to_x
        self.monitor_metric = monitor_metric
        self.lr_scheduler_params = lr_scheduler_params
        self.domain_dim = domain_dim
        self.output_dim = output_dim
        self.loss_name = loss_name

        # whether to use y as input to the encoder
        self.use_y_forward = include_y0_input

        # currently used for including v0 in the input to the encoder
        # can also be used for decoding later on

        self.model = SimpleEncoder(input_dim=input_dim,
                                    output_dim=output_dim,
                                    domain_dim=domain_dim,
                                    d_model=d_model, 
                                    nhead=nhead, 
                                    num_layers=num_layers,
                                    max_sequence_length=max_sequence_length,
                                    do_layer_norm=do_layer_norm,
                                    use_transformer=use_transformer,
                                    use_positional_encoding=use_positional_encoding,
                                    append_position_to_x=append_position_to_x,
                                    pos_enc_coeff=pos_enc_coeff,
                                    include_y0_input=include_y0_input,
                                    activation=activation,
                                    dropout=dropout,
                                    norm_first=norm_first,
                                    dim_feedforward=dim_feedforward)
        
        self.test_losses = {}

    def forward(self, x, y, coords_x, coords_y):
        coords_x = coords_x[0].unsqueeze(2)
        coords_y = coords_y[0].unsqueeze(2)
        if self.first_forward:
            self.first_forward = False
            self.plot_positional_encoding(x, coords_x)

        if self.use_y_forward:
            return self.model(x, y=y, coords_x=coords_x)
        else: 
            return self.model(x, y=None, coords_x=coords_x)

    def training_step(self, batch, batch_idx):
        x, y, coords_x, coords_y = batch
        y_hat = self.forward(x, y, coords_x, coords_y)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/train/mse", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        # Sup norm loss
        loss_sup  = torch.max(torch.abs(y_hat - y))
        self.log("loss/train/sup", loss_sup, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        #Relative L2 loss
        rel_loss = torch.mean(torch.div(torch.sqrt(torch.mean(torch.mean((y_hat -y)**2, dim=1),dim=1)),
                                        torch.sqrt(torch.mean(torch.mean(y**2, dim=1),dim=1))))
        self.log("loss/train/rel_L2", rel_loss, on_step=False,
                 on_epoch=True, prog_bar=True)


        if batch_idx == 0:
            self.make_batch_figs(x, y, y_hat, coords_x, coords_y, tag='Train')

        loss_dict = {"l2": loss, "l2_rel": rel_loss}
        return loss_dict[self.loss_name]

    def on_after_backward(self):
        self.log_gradient_norms(tag='afterBackward')

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer and its gradient
        # If using mixed precision, the gradients are already unscaled here
        self.log_gradient_norms(tag='beforeOptimizer')
        self.log_parameter_norms(tag='beforeOptimizer')

    def log_gradient_norms(self, tag=''):
        norm_type = 2.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.detach().norm(norm_type)
                name = name.replace('.', '_')
                self.log(f"grad_norm/{tag}/{name}", grad_norm,
                         on_step=False, on_epoch=True, prog_bar=False)

    def log_parameter_norms(self, tag=''):
        norm_type = 2.0
        for name, param in self.named_parameters():
            param_norm = param.detach().norm(norm_type)
            name = name.replace('.', '_')
            self.log(f"param_norm/{tag}/{name}", param_norm,
                     on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x, y, coords_x, coords_y = batch
        y_hat = self.forward(x, y, coords_x, coords_y)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/val/mse", loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        # Sup norm loss
        loss_sup  = torch.max(torch.abs(y_hat - y))
        self.log("loss/val/sup", loss_sup, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        #Relative L2 loss

        rel_loss = torch.mean(torch.div(torch.sqrt(torch.mean(torch.mean((y_hat -y)**2, dim=1),dim=1)),
                                        torch.sqrt(torch.mean(torch.mean(y**2, dim=1),dim=1))))
        self.log("loss/val/rel_L2", rel_loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self.make_batch_figs(x, y, y_hat, coords_x, coords_y, tag='Val')

        loss_dict = {"l2": loss, "l2_rel": rel_loss}
        return loss_dict[self.loss_name]

    def plot_positional_encoding(self, x, coords):
        pe = self.model.positional_encoding(x, coords)
        plt.figure()
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plt.imshow(pe.detach().cpu().numpy(), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Positional Encoding')
        wandb.log({f"plots/Positional Encoding": wandb.Image(fig)})
        plt.close()

    def make_batch_figs(self, x, y, y_hat, coords_x, coords_y, tag='', n_examples=5):
        if n_examples > x.shape[0]:
            n_examples = x.shape[0]
        idx = torch.arange(n_examples)
        y_pred = y_hat[idx].detach().cpu().numpy()
        y_true = y[idx].detach().cpu().numpy()
        coords_x = coords_x[idx].detach().cpu().numpy()
        coords_y = coords_y[idx].detach().cpu().numpy()

        if self.domain_dim == 1:
            self.batch_figs_1D(x, y_true, y_pred, coords_x, coords_y, tag, idx)
        elif self.domain_dim == 2:
            self.batch_figs_2D(x, y_true, y_pred, coords_x, coords_y, tag, idx)

    def make_test_figs(self, median_sample, worst_sample, tag=None):

        if self.domain_dim == 1:
            self.test_figs_1D(median_sample, worst_sample, tag)
        elif self.domain_dim == 2:
            self.test_figs_2D(median_sample, worst_sample, tag)

    def batch_figs_1D(self, x, y_true, y_pred, coords_x, coords_y, tag, idx):
        # Plot Trajectories
        plt.figure()
        fig, axs = plt.subplots(
            nrows=y_true.shape[-1], ncols=len(idx), figsize=(10 * len(idx), 6 * y_true.shape[-1]), sharex=True, squeeze=False)

        for col, idx_val in enumerate(idx):
            for i, ax in enumerate(axs[:, col]):
                ax.plot(coords_y[idx_val], y_true[idx_val, :, i], linewidth=3,
                        color='blue', label='Ground Truth')
                ax.plot(coords_y[idx_val], y_pred[idx_val, :, i], linewidth=3,
                        color='red', label='Prediction')
                ax.set_xlabel('Time')
                ax.set_ylabel('Prediction')
                ax.set_title(
                    f'Trajectory for predicted component {i} (Index {idx_val})')
                if col == 0:
                    ax.legend()

        fig.suptitle(f'{tag} Trajectories: Prediction vs. Truth')
        plt.subplots_adjust(hspace=0.5)
        wandb.log(
            {f"plots/{tag}/Trajectories: Prediction vs. Truth": wandb.Image(fig)})
        plt.close()

        # compute value of each encoder layer sequentially
        # choose 3 random hidden dimensions to plot throughout
        idx_dim = [0, 1, 2]
        plt.figure()
        fig, axs = plt.subplots(
            nrows=len(idx_dim), ncols=len(idx), figsize=(10 * len(idx), 6 * len(idx_dim)), sharex=True)

        for col, idx_val in enumerate(idx):
            x_layer_output = self.model.linear_in(x[idx_val])
            for j, id in enumerate(idx_dim):
                axs[j, col].set_title(
                    f'Embedding dimension {id} over layer depth (Index {idx_val})')
                axs[j, col].plot(coords_x[idx_val],
                                x_layer_output.detach().cpu().numpy()[
                                    :, id].squeeze(),
                                linewidth=3, alpha=0.8, label='Layer {}'.format(0),
                                color=plt.cm.viridis(0))
            #need to add batch dimension to x_layer_output for custom implementation of transformer
            x_layer_output = x_layer_output.unsqueeze(0)
            for i, layer in enumerate(self.model.encoder.layers):
                #need to make sure coords_x has dimension (seq_ln,1,1)
                x_layer_output = layer(x_layer_output,torch.tensor(coords_x[idx_val]).to(x_layer_output).unsqueeze(1))
                # Plot the output of this layer
                for j, id in enumerate(idx_dim):
                    axs[j, col].plot(coords_x[idx_val].squeeze(),
                                    x_layer_output.detach().cpu().numpy()[0,
                                        :, id].squeeze(),
                                    linewidth=3, alpha=0.8, label=f'Layer {i+1}',
                                    color=plt.cm.viridis((i+1) / (len(self.model.encoder.layers))))

        axs[0, 0].legend()
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(f'{tag} Evolution of the Encoder Layers')
        wandb.log({f"plots/{tag}/Encoder Layer Plot": wandb.Image(fig)})
        plt.close('all')

    def batch_figs_2D(self, x, y_true, y_pred, coords_x, coords_y, tag, idx, n_grid=100):

        # Each element of y_true and y_pred is a 2D field with coordinates given by coords_y
        # plot the values of y_true and y_pred at the indices given by coords_y

        # Plot a 3 paneled figure with 3 scalar 2-d fields (heatmaps)
        # 1. Ground truth
        # 2. Prediction
        # 3. Relative difference

        # get the low and high indices of the y coordinates
        i_low_1, i_low_2 = np.min(coords_y[...,0]), np.min(coords_y[...,1])
        i_high_1, i_high_2 = np.max(coords_y[...,0]), np.max(coords_y[...,1])
        # build a meshgrid of coordinates based on coords_y
        y1i, y2i = np.meshgrid(np.linspace(i_low_1, i_high_1, n_grid), np.linspace(i_low_2, i_high_2, n_grid))

        # get the low and high indices of the x coordinates
        i_low_1, i_low_2 = np.min(coords_x[...,0]), np.min(coords_x[...,1])
        i_high_1, i_high_2 = np.max(coords_x[...,0]), np.max(coords_x[...,1])
        # build a meshgrid of coordinates based on coords_y
        x1i, x2i = np.meshgrid(np.linspace(i_low_1, i_high_1, n_grid), np.linspace(i_low_2, i_high_2, n_grid))

        plt.figure()
        fig, axs = plt.subplots(
            nrows=4, ncols=len(idx), figsize=(10 * len(idx), 6 * 4), sharex=True, squeeze=False)

        for col, idx_val in enumerate(idx):
            x_input_i = griddata(
                (coords_x[idx_val, :, 0], coords_x[idx_val, :, 1]), x[idx_val].detach().cpu().numpy(), (x1i, x2i), method='linear')
            y_true_i = griddata(
                (coords_y[idx_val, :, 0], coords_y[idx_val, :, 1]), y_true[idx_val], (y1i, y2i), method='linear')
            y_pred_i = griddata((coords_y[idx_val, :, 0], coords_y[idx_val, :, 1]), y_pred[idx_val], (y1i, y2i), method='linear')
            #y_true_i_norm = np.sqrt((1/(y_true_i.shape[0]*y_true_i.shape[1]))*np.sum(y_true_i**2))
            y_rel_diff_i = np.abs(y_pred_i - y_true_i) #/ np.abs(y_true_i + 1e-5)
            #y_rel_diff_i = np.abs(y_pred_i - y_true_i)

            #plot median and worst case relative error

            for i, ax in enumerate(axs[:, col]):
                if i == 0:
                    # plot input field x
                    im = ax.imshow(x_input_i, cmap='viridis')
                    ax.set_title(
                        f'Input Field (Index {idx_val})')

                if i == 1:
                    im = ax.imshow(y_true_i, cmap='viridis')
                    ax.set_title(
                        f'Ground Truth (Index {idx_val})')
                elif i == 2:
                    im = ax.imshow(y_pred_i, cmap='viridis')
                    ax.set_title(
                        f'Prediction (Index {idx_val})')
                elif i == 3:
                    # plot absolute relative error in log scale (difference divided by ground truth)
                    #im = ax.imshow(np.log10(y_rel_diff_i + 1e-10), cmap='viridis', vmin=-5, vmax=3)
                    im = ax.imshow(np.log10(y_rel_diff_i), cmap='inferno', vmin=-7,vmax=1)
                    ax.set_title(
                        f'Pointwise Error (Index {idx_val})')
                fig.colorbar(im, ax=ax)

        fig.suptitle(f'{tag} Predicted Fields: Prediction vs. Truth')
        plt.subplots_adjust(hspace=0.5)
        wandb.log(
            {f"plots/{tag}/Predicted Fields: Prediction vs. Truth": wandb.Image(fig)})
        plt.close()

    def test_step(self, batch, batch_idx, dataloader_idx=0):

        dt = self.trainer.datamodule.test_sample_rates[dataloader_idx]

        x, y, coords_x, coords_y = batch
        y_hat = self.forward(x, y, coords_x, coords_y)
        loss = F.mse_loss(y_hat, y)
        self.log(f"loss/test/mse/dt{dt}", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        # Sup norm loss
        loss_sup  = torch.max(torch.abs(y_hat - y))
        self.log(f"loss/test/sup/dt{dt}", loss_sup, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        #Relative L2 loss

        rel_loss = torch.mean(torch.div(torch.sqrt(torch.mean(torch.mean((y_hat -y)**2, dim=1),dim=1)),
                                        torch.sqrt(torch.mean(torch.mean(y**2, dim=1),dim=1))))
        self.log(f"loss/test/rel_L2/dt{dt}", rel_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        # Save the test losses and corresponding indices for each DataLoader index
        if dataloader_idx not in self.test_losses:
            self.test_losses[dataloader_idx] = {'losses': [], 'indices': [], 'prediction':[]}
        
        for i in range(x.size(0)):
            loss_sample = torch.div(torch.sqrt(torch.mean(torch.mean((y_hat[i:i+1] -y[i:i+1])**2, dim=1),dim=1)),
                                        torch.sqrt(torch.mean(torch.mean(y[i:i+1]**2, dim=1),dim=1)))

            # Save the test losses and corresponding indices for each DataLoader index
            self.test_losses[dataloader_idx]['losses'].append(loss_sample.cpu().detach().numpy())
            self.test_losses[dataloader_idx]['indices'].append((batch_idx, i))
            self.test_losses[dataloader_idx]['prediction'].append(y_hat[i:i+1])


        loss_dict = {"l2": loss, "l2_rel": rel_loss}
        return loss_dict[self.loss_name]
    

    def test_figs_1D(self, median_sample, worst_sample, tag):


        x_median, y_median, coords_x_median, coords_y_median,y_pred_median, median_error = median_sample
        x_min, y_min, coords_x_min, coords_y_min, y_pred_min, min_error = worst_sample

        #make all the previous into numpy arrays
        x_median, y_median, coords_x_median, coords_y_median, y_pred_median, median_error = x_median.cpu().numpy(), y_median.cpu().numpy(), coords_x_median.cpu().numpy(), coords_y_median.cpu().numpy(), y_pred_median.cpu().numpy(), median_error[0]
        x_min, y_min, coords_x_min, coords_y_min , y_pred_min, min_error = x_min.cpu().numpy(), y_min.cpu().numpy(), coords_x_min.cpu().numpy(), coords_y_min.cpu().numpy() , y_pred_min.cpu().numpy(), min_error[0]

        plt.figure()
        fig, axs = plt.subplots(
            nrows=(self.output_dim+1),
            ncols=2,
            figsize=(14 * y_min.shape[1], 7 * self.output_dim),
            sharex=True,
            squeeze=False,
        )

        axs[0, 1].set_title(
                f"Trajectories for predicted components: \nMaximum Relative L2 Error ({min_error:.2e})"
            )

        axs[0, 0].set_title(
            f"Trajectories for predicted components: \nMedian Relative L2 Error ({median_error:.2e})"
        )

        # In the first column, plot the inputs

        # Min
        axs[0, 1].plot(
            coords_x_min[:, 0],
            x_min[:, 0],
            linewidth=3,
            color="blue",
            label=f"Input",
        )
        axs[0, 1].set_xlabel("Time")
        axs[0, 1].legend()

        # Median
        axs[0, 0].plot(
            coords_x_median[:, 0],
            x_median[:, 0],
            linewidth=3,
            color="blue",
            label=f"Input",
        )
        axs[0, 0].set_xlabel("Time")
        axs[0, 0].legend()

        for i in range(self.output_dim):

            # Min
            axs[i+1, 1].plot(
                coords_y_min[:, 0],
                y_min[:, i],
                linewidth=3,
                color="blue",
                label="Ground Truth",
            )
            axs[i+1, 1].plot(
                coords_y_min[:, 0],
                y_pred_min[0, :, i],
                linewidth=3,
                color="red",
                label="Prediction",
            )
            axs[i+1, 1].set_xlabel("Time")
            axs[i+1, 1].set_ylabel("Prediction")
            axs[i+1, 1].legend()

            # Median
            axs[i+1, 0].plot(
                coords_y_median[:, 0],
                y_median[:, i],
                linewidth=3,
                color="blue",
                label="Ground Truth",
            )
            axs[i+1, 0].plot(
                coords_y_median[:, 0],
                y_pred_median[0, :, i],
                linewidth=3,
                color="red",
                label="Prediction",
            )
            axs[i+1, 0].set_xlabel("Time")
            axs[i+1, 0].set_ylabel("Prediction")
            axs[i+1, 0].legend()

            # If you want to add space between rows, you can adjust the hspace parameter here.

        fig.suptitle(f"Trajectories: Prediction vs. Truth (sample rate = {tag})")
        plt.subplots_adjust(hspace=0.5)
        wandb.log(
                {f"plots/{tag}/Predicted Fields: Prediction vs. Truth": wandb.Image(fig)})
        plt.close()


    def test_figs_2D(self, median_sample, worst_sample, tag):


        x_median, y_median, coords_x_median, coords_y_median,y_pred_median, median_error = median_sample
        x_min, y_min, coords_x_min, coords_y_min, y_pred_min, min_error = worst_sample

        #make all the previous into numpy arrays
        x_median, y_median, coords_x_median, coords_y_median, y_pred_median, median_error = x_median.cpu().numpy(), y_median.cpu().numpy(), coords_x_median.cpu().numpy(), coords_y_median.cpu().numpy(), y_pred_median.cpu().numpy(), median_error[0]
        x_min, y_min, coords_x_min, coords_y_min , y_pred_min, min_error = x_min.cpu().numpy(), y_min.cpu().numpy(), coords_x_min.cpu().numpy(), coords_y_min.cpu().numpy() , y_pred_min.cpu().numpy(), min_error[0]

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
            nrows=2, ncols=4, figsize=(24, 12), sharex=True, squeeze=False)

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
            (0, 0): x_input_median,
            (0, 1): y_true_median,
            (0, 2): y_pred_median,
            (0, 3): np.log10(y_rel_diff_median),
            (1, 0): x_input_min,
            (1, 1): y_true_min,
            (1, 2): y_pred_min,
            (1, 3): np.log10(y_rel_diff_min)
        }

        for (i, j), data in data_sets.items():
            bounds = (-7, 1) if j == 3 else (None, None)
            im = axs[i, j].imshow(data, cmap='viridis' if j < 3 else 'inferno', vmin=bounds[0], vmax=bounds[1])
            error_info = f"Median Relative L2 Error ({median_error:.2e})" if i == 0 else f"Maximum Relative L2 Error ({min_error:.2e})"
            axs[i, j].set_title(
                f"{'Input Field: ' if j == 0 else 'Ground Truth:' if j == 1 else 'Prediction:' if j == 2 else 'Pointwise Error:'}\n "
                f"{error_info}", 
                fontsize=14
            )

            fig.colorbar(im, ax=axs[i, j], shrink=0.6)


        fig.suptitle(f'{tag} Predicted Fields: Prediction vs. Truth', fontsize=30)
        plt.subplots_adjust(hspace=-0.1, wspace=0.1)
        wandb.log(
                {f"plots/{tag}/Predicted Fields: Prediction vs. Truth": wandb.Image(fig)})
        plt.close()
    
    def on_test_epoch_end(self):

        for dataloader_idx, dataloader_losses in self.test_losses.items():
            losses = torch.tensor(dataloader_losses['losses'])

            # Calculate median and worst error indices
            median_idx = int(torch.argsort(losses,dim=0)[len(losses) // 2])
            worst_idx = int(torch.argmax(losses))

            # Access indices of median and worst cases
            median_batch_idx, median_sample_idx = dataloader_losses['indices'][median_idx]
            worst_batch_idx, worst_sample_idx = dataloader_losses['indices'][worst_idx]

            # Access the test data loaders
            dt = self.trainer.datamodule.test_sample_rates[dataloader_idx]
            test_dataloader = self.trainer.datamodule.test_dataloader()[dt]

            # Retrieve the samples using indices, 
            ###################################################
            ###################################################
            # !!!!!terrible code!!!!!
            for batch_idx, batch in enumerate(test_dataloader):
                if batch_idx == median_batch_idx:
                    median_batch = batch
                if batch_idx == worst_batch_idx:
                    worst_batch = batch


            median_sample = [median_batch[i][median_sample_idx] for i in range(4)]+[dataloader_losses['prediction'][median_idx], dataloader_losses['losses'][median_idx]]
            worst_sample =  [worst_batch[i][worst_sample_idx] for i in range(4)]+[dataloader_losses['prediction'][worst_idx], dataloader_losses['losses'][worst_idx]]
            ###################################################
            ###################################################

            # Now you can use median_sample and worst_sample to plot relevant information or perform further analysis
    
            self.make_test_figs(median_sample, worst_sample, tag=f'Test/dt{dt}')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        config = {
            # REQUIRED: The scheduler instance
            "scheduler": ReduceLROnPlateau(optimizer, verbose=True, **self.lr_scheduler_params),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.monitor_metric,  # "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": config,
        }

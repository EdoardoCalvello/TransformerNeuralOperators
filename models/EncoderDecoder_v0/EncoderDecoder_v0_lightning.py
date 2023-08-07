import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt

from models.SimpleEncoder.SimpleEncoder_pytorch import SimpleEncoder

# Define the pytorch lightning module for training the Simple Encoder model
class SimpleEncoderModule(pl.LightningModule):
    def __init__(self, input_dim=1, output_dim=1, d_model=32, nhead=8, num_layers=6,
                 learning_rate=0.01, max_sequence_length=100,
                 do_layer_norm=True,
                 use_transformer=True,
                 use_positional_encoding=True,
                 activation='relu',
                 monitor_metric='train_loss',
                 lr_scheduler_params={'patience': 3,
                                      'factor': 0.5},
                 dropout=0.1, norm_first=False, dim_feedforward=2048):
        super(SimpleEncoderModule, self).__init__()
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.use_transformer = use_transformer
        self.use_positional_encoding = use_positional_encoding
        self.monitor_metric = monitor_metric
        self.lr_scheduler_params = lr_scheduler_params

        self.model = SimpleEncoder(input_dim=input_dim,
                                    output_dim=output_dim, 
                                    d_model=d_model, 
                                    nhead=nhead, 
                                    num_layers=num_layers,
                                    max_sequence_length=max_sequence_length,
                                    do_layer_norm=do_layer_norm,
                                    use_transformer=use_transformer,
                                    use_positional_encoding=use_positional_encoding,
                                    activation=activation,
                                    dropout=dropout,
                                    norm_first=norm_first,
                                    dim_feedforward=dim_feedforward)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        # Sup norm loss
        loss_sup  = torch.max(torch.abs(y_hat - y))
        self.log("train_loss_sup", loss_sup, on_step=False,
                 on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self.make_batch_figs(x, y, y_hat, tag='Train')

        return loss

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
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Sup norm loss
        loss_sup  = torch.max(torch.abs(y_hat - y))
        self.log("val_loss_sup", loss_sup, on_step=False,
                 on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self.make_batch_figs(x, y, y_hat, tag='Val')
        return loss

    def make_batch_figs(self, x, y, y_hat, tag=''):
        idx = [0]  # plot only the first element of the batch
        y_pred = y_hat[idx].detach().cpu().numpy()
        y_true = y[idx].detach().cpu().numpy()

        # time domain
        x_arange = torch.arange(0, x.shape[1], 1)

        plt.figure()
        fig, axs = plt.subplots(
            nrows=y_true.shape[-1], ncols=1, figsize=(10, 6), sharex=True, squeeze=False)
        for i, ax in enumerate(axs):
            ax = ax[0] # squeeze
            ax.plot(x_arange.detach().cpu().numpy(), y_true[:,:,i].squeeze(), linewidth=3,
                        color='blue', label='Ground Truth')
            ax.plot(x_arange.detach().cpu().numpy(), y_pred[:,:,i].squeeze(), linewidth=3,
                        color='red', label='Prediction')
            ax.set_xlabel('Time')
            ax.set_ylabel('Prediction')
            ax.set_title('Trajectory for predicted component {}'.format(i))
        axs[0][0].legend() # only put legend on first plot
        plt.grid(True)
        fig.suptitle(f'{tag} Trajectories: Prediction vs. Truth')
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(f"traj_plot_{tag}.png")
        wandb.log(
            {f"{tag} Trajectories: Prediction vs. Truth": wandb.Image(f"traj_plot_{tag}.png")})
        plt.close()
        os.remove(f"traj_plot_{tag}.png")

        # compute value of each encoder layer sequentially
        # choose 3 random hidden dimensions to plot throughout
        idx_dim = [0, 1, 2]
        plt.figure()
        fig, axs = plt.subplots(
            nrows=len(idx_dim), ncols=1, figsize=(10, 6), sharex=True)

        x_layer_output = self.model.linear_in(x[idx])
        for j, id in enumerate(idx_dim):
            axs[j].set_title(
                'Embedding dimension {} over layer depth'.format(id))
            axs[j].plot(x_arange,
                        x_layer_output.detach().cpu().numpy()[
                            :, :, id].squeeze(),
                        linewidth=3, alpha=0.8, label='Layer {}'.format(0),
                        color=plt.cm.viridis(0))
        for i, layer in enumerate(self.model.encoder.layers):
            x_layer_output = layer(x_layer_output)
            # Plot the output of this layer
            for j, id in enumerate(idx_dim):
                axs[j].plot(x_arange,
                            x_layer_output.detach().cpu().numpy()[
                                :, :, id].squeeze(),
                            linewidth=3, alpha=0.8, label=f'Layer {i+1}',
                            color=plt.cm.viridis((i+1) / (len(self.model.encoder.layers))))

        axs[0].legend()
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(f'{tag} Evolution of the Encoder Layers')
        plt.savefig("encoder_layer_plot.png")
        wandb.log({f"{tag} Encoder Layer Plot": wandb.Image(
            "encoder_layer_plot.png")})
        os.remove("encoder_layer_plot.png")
        plt.close('all')


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        # Sup norm loss
        loss_sup  = torch.max(torch.abs(y_hat - y))
        self.log("test_loss_sup", loss_sup, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        return loss

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
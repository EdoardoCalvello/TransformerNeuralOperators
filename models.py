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

# Define the neural network model
class TransformerEncoder(pl.LightningModule):
    def __init__(self, input_dim=1, output_dim=1, d_model=32, nhead=8, num_layers=6,
                 learning_rate=0.01, max_sequence_length=100,
                 use_transformer=True,
                 use_positional_encoding=True,
                 activation='relu',
                 monitor_metric='train_loss',
                 lr_scheduler_params={'patience': 3,
                                      'factor': 0.5},
                 dropout=0.1, norm_first=False, dim_feedforward=2048):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.use_transformer = use_transformer
        self.use_positional_encoding = use_positional_encoding
        self.monitor_metric = monitor_metric
        self.lr_scheduler_params = lr_scheduler_params

        self.set_positional_encoding()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            dim_feedforward=dim_feedforward,
            batch_first=True)  # when batch first, expects input tensor (batch_size, Seq_len, input_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        # (Seq_len,batch_size,input_dim) if batch_first=False or (N, S, E) if batch_first=True.
        # where S is the source sequence length, N is the batch size, E is the feature number, T is the target sequence length,

        self.linear_in = nn.Linear(input_dim, d_model)
        self.linear_out = nn.Linear(d_model, output_dim)

    def set_positional_encoding(self):
        pe = torch.zeros(self.max_sequence_length, self.d_model)
        position = torch.arange(
            0, self.max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, self.d_model, 2, dtype=torch.float) * -(torch.log(torch.tensor(10000.0)) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add a batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape) # (batch_size, seq_len, dim_state)
        # x = x.permute(1,0,2) # (seq_len, batch_size, dim_state)
        x = self.linear_in(x)  # (batch_size, seq_len, input_dim)

        if self.use_positional_encoding:
            x = x + self.pe[:, :x.size(1)]  # (batch_size, seq_len, dim_state)

        if self.use_transformer:
            x = self.encoder(x)  # (batch_size, seq_len, dim_state)

        x = self.linear_out(x)  # (seq_len, batch_size, output_dim)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
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

        if batch_idx == 0:
            idx = [0] # plot only the first trajectory
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
            fig.suptitle('Trajectories: Prediction vs. Truth')
            plt.subplots_adjust(hspace=0.5)
            plt.savefig("traj_plot.png")
            wandb.log({"Trajectories: Prediction vs. Truth": wandb.Image("traj_plot.png")})
            plt.close()
            os.remove("traj_plot.png")

            # compute value of each encoder layer sequentially
            # choose 3 random hidden dimensions to plot throughout
            idx_dim = [0, 1, 2]
            plt.figure()
            fig, axs = plt.subplots(
                nrows=len(idx_dim), ncols=1, figsize=(10, 6), sharex=True)

            x_layer_output = self.linear_in(x[idx])
            for j, id in enumerate(idx_dim):
                axs[j].set_title(
                    'Embedding dimension {} over layer depth'.format(id))
                axs[j].plot(x_arange,
                            x_layer_output.detach().cpu().numpy()[
                                :, :, id].squeeze(),
                            linewidth=3, alpha=0.8, label='Layer {}'.format(0),
                            color=plt.cm.viridis(0))
            for i, layer in enumerate(self.encoder.layers):
                x_layer_output = layer(x_layer_output)
                # Plot the output of this layer
                for j, id in enumerate(idx_dim):
                    axs[j].plot(x_arange,
                                x_layer_output.detach().cpu().numpy()[
                                    :, :, id].squeeze(),
                                linewidth=3, alpha=0.8, label=f'Layer {i+1}',
                                color=plt.cm.viridis((i+1) / (len(self.encoder.layers))))

            axs[0].legend()
            plt.subplots_adjust(hspace=0.5)
            fig.suptitle('Evolution of the Encoder Layers')
            plt.savefig("encoder_layer_plot.png")
            wandb.log({"Encoder Layer Plot": wandb.Image(
                "encoder_layer_plot.png")})
            os.remove("encoder_layer_plot.png")
            plt.close('all')

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False,
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

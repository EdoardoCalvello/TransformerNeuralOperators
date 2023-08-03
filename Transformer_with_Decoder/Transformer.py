import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb


from MultiHeadAttention import MultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward
from PositionalEncoding import PositionalEncoding

import pdb


class EncoderLayer(pl.LightningModule):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(pl.LightningModule):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(pl.LightningModule):
    def __init__(self, input_dim=1, output_dim=1, d_model=32, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=100, dropout=0.1,learning_rate=0.01):
        super(Transformer, self).__init__()

        self.max_seq_length = max_seq_length
        #using linear for embedding
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.decoder_embedding = nn.Linear(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, self.max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.learning_rate = learning_rate
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask


    def forward(self, src, tgt=None):
        src_mask, tgt_mask = self.generate_mask(src, tgt) if tgt is not None else (None, None)

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded

        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, None)  # src_mask

        if tgt is not None:
            tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
            dec_output = tgt_embedded

            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, None, None)  # src_mask, tgt_mask

            output = self.fc(dec_output)
            return output
        else:
            # Autoregressive decoding during validation and testing
            # Start with a dummy token as the initial input to the decoder
            dummy_input = torch.zeros_like(src_embedded[:, :1, :])  # Shape: (batch_size, 1, d_model) #src_embedded[:, :1, :]?
            dec_output = dummy_input
            
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, None, None)  # src_mask, tgt_mask

            # Generate the output sequence using the decoder output
            output_seq = [dec_output[:, -1:, :]]  # Store the first prediction (dummy_input)
            for _ in range(self.max_seq_length - 1):
                for dec_layer in self.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, None, None)  # src_mask, tgt_mask
                output_seq.append(dec_output[:, -1:, :])
                dec_output = torch.cat(output_seq, dim=1) #[1:]

            # Concatenate all the generated decoder outputs along the sequence length dimension
            output_seq = torch.cat(output_seq, dim=1)  # Shape: (batch_size, max_seq_length, d_model)

            output = self.fc(output_seq)
            return output



    #def forward(self, src, tgt):

     #   src_mask, tgt_mask = self.generate_mask(src, tgt)
       
     #   src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
     #   tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        #src_embedded = self.dropout((self.encoder_embedding(src)))
        #tgt_embedded = self.dropout((self.decoder_embedding(tgt)))

     #   enc_output = src_embedded
     #   for enc_layer in self.encoder_layers:
     #       enc_output = enc_layer(enc_output, None) #src_mask

     #   dec_output = tgt_embedded
     #   for dec_layer in self.decoder_layers:
     #       dec_output = dec_layer(dec_output, enc_output, None, None) #src_mask, tgt_mask

     #   output = self.fc(dec_output)
     #   return output
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x,y)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)    
        return loss

    def on_after_backward(self):
        self.log_gradient_norms(tag='afterBackward')

    '''
    def on_before_optimizer_step(self):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        self.log_gradient_norms(tag='beforeOptimizer')
    '''

    def log_gradient_norms(self, tag=''):
        norm_type = 2.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.detach().norm(norm_type)
                name = name.replace('.', '_')
                self.log(f"grad_norm/{tag}/{name}", grad_norm,
                         on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x,None)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            num_points = 1
            idx = torch.randint(0, len(y), (num_points,))
            y_pred = y_hat[idx].detach().cpu().numpy().flatten()
            y_true = y[idx].detach().cpu().numpy().flatten()

            plt.figure(figsize=(10, 6))
            plt.scatter(x[idx].detach().cpu().numpy(), y_true,
                        color='blue', label='Ground Truth')
            plt.scatter(x[idx].detach().cpu().numpy(), y_pred,
                        color='red', label='Prediction')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.title('Prediction vs. Truth')
            plt.grid(True)

            plt.savefig("scatter_plot.png")
            wandb.log({"Prediction vs. Truth": wandb.Image("scatter_plot.png")})
            plt.close()
            os.remove("scatter_plot.png")

        return loss

    def test_step(self, batch):
        x, y = batch
        y_hat = self.forward(x,None)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        config = {
            # REQUIRED: The scheduler instance
            "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "train_loss",  # "val_loss",
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
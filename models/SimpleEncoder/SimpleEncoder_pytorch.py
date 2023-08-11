import os
import torch
import torch.nn as nn

# installed pytorch
# from torch.nn import TransformerEncoder
# from torch.nn import TransformerEncoderLayer

# source code
from models.pytorch_transformer_custom import TransformerEncoder 
from models.pytorch_transformer_custom import TransformerEncoderLayer

# Define the neural network model
class SimpleEncoder(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1, d_model=32, nhead=8, num_layers=6,
                 learning_rate=0.01, max_sequence_length=100,
                 do_layer_norm=True,
                 use_transformer=True,
                 use_positional_encoding='continuous',
                 include_y0_input=False,
                 activation='relu',
                 dropout=0.1, norm_first=False, dim_feedforward=2048):
        super(SimpleEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.use_transformer = use_transformer
        self.use_positional_encoding = use_positional_encoding
        self.include_y0_input = include_y0_input # whether to use y as input to the encoder

        self.set_positional_encoding()

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            do_layer_norm=do_layer_norm,
            dim_feedforward=dim_feedforward,
            batch_first=True)  # when batch first, expects input tensor (batch_size, Seq_len, input_dim)
        self.encoder = TransformerEncoder(
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
        self.register_buffer('pe_discrete', pe)

        # for continuous time positional encoding
        even_inds = torch.arange(0, self.d_model, 2).unsqueeze(0)
        odd_inds = torch.arange(1, self.d_model, 2).unsqueeze(0)
        self.register_buffer('even_inds', even_inds)
        self.register_buffer('odd_inds', odd_inds)

    def pe_continuous(self, times):
        '''apply a positional encoding to sequence x evaluated at times t'''
        pe = torch.zeros(times.shape[0], self.d_model)
        pe[:, 0::2] = torch.sin(100 * times * 10**(-4 * self.even_inds / self.d_model))
        pe[:, 1::2] = torch.cos(100 * times * 10**(-4 * self.odd_inds / self.d_model))
        return pe

    def positional_encoding(self, x, times):
        # x: (batch_size, seq_len, input_dim)
        # pe: (1, seq_len, d_model)
        # x + pe[:, :x.size(1)]  # (batch_size, seq_len, d_model)
        if self.use_positional_encoding=='discrete':
            pe = self.pe_discrete[:, :x.size(1)]
        elif self.use_positional_encoding=='continuous':
            pe = self.pe_continuous(times)
        else: # no positional encoding
            pe = torch.tensor(0)

        return pe

    def apply_positional_encoding(self, x, times):
        return x + self.positional_encoding(x, times)

    def forward(self, x, y=None, times=None):
        if self.include_y0_input:
            #this only works when the input dimension is 1, indeed how would you concatenate initial condition with the input otherwise?
            # x = x.permute(1,0,2) # (seq_len, batch_size, dim_state)
            # reshape to make sure we have dim (batch,1,dim_output)
            initial_cond = y[:, 0:1, :].permute(0, 2, 1)  # (batch_size, dim_state, 1)
            # (batch_size, seq_len+output_dim, dim_state)
            x = torch.cat((initial_cond, x), dim=1)

        x = self.linear_in(x)  # (batch_size, seq_len, input_dim)

        # times = torch.linspace(0, 1, x.shape[1]).unsqueeze(1)
        # can use first time because currently all batches share the same time discretization
        x = self.apply_positional_encoding(x, times)

        if self.use_transformer:
            x = self.encoder(x)  # (batch_size, seq_len, dim_state)

        x = self.linear_out(x)  # (seq_len, batch_size, output_dim)

        if self.include_y0_input:
            return x[:, self.output_dim:, :]
        else:
            return x
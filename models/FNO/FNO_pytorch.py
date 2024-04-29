import torch
import torch.nn as nn
from torch.nn import functional as F
import copy

from models.transformer_custom import SpectralConv2d


class FourierLayer(nn.Module):

    def __init__(self, d_model=32, dropout=0.1, activation='relu', modes=None, im_size=None):
        super(FourierLayer, self).__init__()

        self.d_model = d_model
        self.dropout = dropout
        self.modes1 = modes[0]
        self.modes2 = modes[1]
        self.im_size = im_size

        self.conv = SpectralConv2d(d_model, d_model, modes[0], modes[1])
        self.linear = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, x):
        # x: (batch_size, d_model, rows, cols)

        x = self.conv(x)
        x = x + self.linear(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.activation(x)
        x = self.norm(x)

        return x

class FNO_Block(nn.Module):
    
    def __init__(self, fourier_layer, num_layers):
        super(FNO_Block, self).__init__()

        self.layers = nn.ModuleList([copy.deepcopy(fourier_layer) for _ in range(num_layers)])

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x


# Define the neural network model
class FNO(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1, domain_dim=1, d_model=32, num_layers=4,
                 learning_rate=0.01,
                 modes=None,
                 im_size=None,
                 activation='relu',
                 dropout=0.1):
        super(FNO, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.domain_dim = domain_dim
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.im_size = im_size


        fourier_layer = FourierLayer(
            d_model=d_model,
            dropout=dropout,
            activation=activation,
            modes=modes,
            im_size=im_size)  # when batch first, expects input tensor (batch_size, Seq_len, input_dim)
        
        self.fno = FNO_Block(fourier_layer, num_layers)
        # (Seq_len,batch_size,input_dim) if batch_first=False or (N, S, E) if batch_first=True.
        # where S is the source sequence length, N is the batch size, E is the feature number, T is the target sequence length,


        self.linear_in = nn.Linear(3,d_model)
        self.size_row = im_size
        self.size_col = im_size 
        self.linear_out = nn.Linear(d_model,1)
            

    def set_im_size(self, new_im_size, new_patch_size):
        # Update the im_size attribute
        self.patch_size = new_patch_size
        self.im_size = new_im_size
        self.size_row = new_im_size
        self.size_col = new_im_size
        self.num_patches = (self.size_row*self.size_col)//(self.patch_size**2)

        #for encoder_layer in self.encoder.layers:
            #encoder_layer.self_attn.scaled_dot_product_attention.scale[0]=torch.tensor(new_im_size**2)
            #encoder_layer.self_attn.im_size=new_im_size


    def forward(self, x, y=None, coords_x=None, coords_y=None, x_train_fourier_normalizer=None):


        residual = x.to(torch.float32)
        x = torch.unsqueeze(x, 1) # (batch_size, 1, rows, cols)
        coords_x = coords_x.squeeze(2) #(domain_dim, rows, cols)
        coords_x = coords_x.repeat(x.shape[0],1,1,1) # (batch_size, domain_dim, rows, cols)
        #addition of coordinates to input as channels
        x = torch.cat((x,coords_x), dim=1) # (batch_size, 1 or 2+domain_dim), rows, cols)
        x = x.to(torch.float32)

        x = x.permute(0,2,3,1)
        x = self.linear_in(x) 
        x = x.permute(0,3,1,2)

        x = self.fno(x)

        x = x.permute(0,2,3,1)
        x = self.linear_out(x)
        x = x.squeeze(3)

        #learning residual
        #x = x + residual
    
        return x
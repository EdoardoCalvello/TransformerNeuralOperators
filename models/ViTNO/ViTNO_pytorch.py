import torch
import torch.nn as nn

from models.transformer_custom import TransformerEncoder_Operator
from models.transformer_custom import TransformerEncoderLayer_ViTNO
from models.transformer_custom import SpectralConv2d, SpectralConv2d_in



# Define the neural network model
class SimpleEncoder(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1, domain_dim=1, d_model=32, nhead=8, num_layers=6,
                 learning_rate=0.01, max_sequence_length=100,
                 do_layer_norm=True,
                 use_transformer=True,
                 patch=False,
                 patch_size=None,
                 modes=None,
                 im_size=None,
                 smoothing = False,
                 smoothing_modes = None,
                 activation='relu',
                 dropout=0.1, norm_first=False, dim_feedforward=2048):
        super(SimpleEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.domain_dim = domain_dim
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.use_transformer = use_transformer
        self.patch = patch
        self.patch_size = patch_size
        #one 2* for imaginary part of fourier transform, one +1 for domain_dim
        self.patch_dim =  (self.domain_dim+1)*(self.patch_size**2)

        self.im_size = im_size
        self.smoothing = smoothing
        self.smoothing_modes = smoothing_modes



        encoder_layer = TransformerEncoderLayer_ViTNO(
            d_model=d_model, nhead=nhead,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            do_layer_norm=do_layer_norm,
            dim_feedforward=dim_feedforward,
            patch_size=patch_size,
            modes=modes,
            im_size=im_size,
            batch_first=True)  # when batch first, expects input tensor (batch_size, Seq_len, input_dim)
        self.encoder = TransformerEncoder_Operator(
            encoder_layer, num_layers=num_layers)
        # (Seq_len,batch_size,input_dim) if batch_first=False or (N, S, E) if batch_first=True.
        # where S is the source sequence length, N is the batch size, E is the feature number, T is the target sequence length,

        self.linear_in = SpectralConv2d_in(3,d_model, modes[0], modes[1])
    

        self.size_row = self.im_size
        self.size_col = self.im_size
        self.linear_out = nn.Linear(d_model,1)
        self.smoothing = SpectralConv2d(1,1,smoothing_modes,smoothing_modes)
            
        self.num_patches = (self.size_row*self.size_col)//(self.patch_size**2)

            

    def set_im_size(self, new_im_size,new_patch_size):
        # Update the im_size attribute
        self.patch_size = new_patch_size
        self.im_size = new_im_size
        self.size_row = new_im_size
        self.size_col = new_im_size
        self.num_patches = (self.size_row*self.size_col)//(self.patch_size**2)

        for encoder_layer in self.encoder.layers:
            encoder_layer.self_attn.scaled_dot_product_attention.scale[0]=torch.tensor(new_im_size**2)
    
    
    def chop_to_patches(self,x):


        num_channels = x.shape[1]
        # - shape is now (batch_size, num_channels, num_patch_per_row, patch_size, num_patch_per_col, patch_size)
        chopped_x = x.reshape(x.shape[0], num_channels, self.size_row // self.patch_size, self.patch_size,
                      self.size_col // self.patch_size, self.patch_size)
        
        # - shape is now (batch_size, num_patch_per_row, num_patch_per_col, num_channels, patch_size, patch_size)
        chopped_x = torch.permute(chopped_x, (0,2,4,1,3,5))
        chopped_x = torch.permute(chopped_x, (0,3,4,5,1,2))
        # - shape is now (batch_size, num_channels, patch_size, patch_size, num_patch_per_row, num_patch_per_col)
        chopped_x = chopped_x.reshape(chopped_x.shape[0], chopped_x.shape[1], chopped_x.shape[2], chopped_x.shape[3], chopped_x.shape[4]*chopped_x.shape[5])
        # - shape is now (batch_size, num_channels, patch_size, patch_size, num_patches)
        chopped_x = torch.permute(chopped_x, (0,4,2,3,1))
        # - shape is now (batch_size, num_patches, patch_size, patch_size, num_channels)

        return chopped_x


    def forward(self, x, y=None, coords_x=None, coords_y=None, x_train_fourier_normalizer=None):


        residual = x.to(torch.float32)
        x = torch.unsqueeze(x, 1) # (batch_size, 1, rows, cols)
        coords_x = coords_x.squeeze(2) #(domain_dim, rows, cols)
        coords_x = coords_x.repeat(x.shape[0],1,1,1) # (batch_size, domain_dim, rows, cols)
        #addition of coordinates to input as channels
        x = torch.cat((x,coords_x), dim=1) # (batch_size, 1 or 2+domain_dim), rows, cols)
        x = self.chop_to_patches(x)
        x = x.to(torch.float32)

        #need to reshape if we want linear_in to actually be spectral conv2d
        x = x.permute(0,1,4,2,3)
        x = self.linear_in(x)  
        x = x.permute(0,1,3,4,2)

        if self.use_transformer:
            x = self.encoder(x)  # (batch_size, seq_len, dim_state)
    
        x = x.reshape(x.shape[0],self.im_size//self.patch_size, self.im_size//self.patch_size ,self.patch_size,self.patch_size, self.d_model)
        x = x.permute(0,1,3,2,4,5)
        x = x.reshape(x.shape[0],self.size_row,self.size_col,self.d_model)

        x = self.linear_out(x)  # (batch_size, seq_len, output_dim)? I think it's (batch_size, seq_len, output_dim)
        x = x.squeeze(3)
        
        if self.smoothing:
            x = x.unsqueeze(1)
            x = x + self.smoothing(x)
            x = x.squeeze(1)

        return x
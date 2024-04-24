import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.fft import rfft2, irfft2


from models.transformer_custom import TransformerEncoder_Operator
from models.transformer_custom import TransformerEncoderLayer_Conv_E
from models.transformer_custom import SpectralConv2d, Smoothing2d



# Define the neural network model
class SimpleEncoder(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1, domain_dim=1, d_model=32, nhead=8, num_layers=6,
                 learning_rate=0.01, max_sequence_length=100,
                 do_layer_norm=True,
                 use_transformer=True,
                 use_positional_encoding='continuous',
                 append_position_to_x=False,
                 patch=False,
                 patch_size=None,
                 fourier = False,
                 modes=None,
                 im_size=None,
                 pos_enc_coeff=2,
                 include_y0_input=False,
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
        self.use_positional_encoding = use_positional_encoding
        self.append_position_to_x = append_position_to_x
        self.pos_enc_coeff = pos_enc_coeff # coefficient for positional encoding
        self.include_y0_input = include_y0_input # whether to use y as input to the encoder
        self.patch = patch
        self.fourier = fourier
        self.patch_size = patch_size
        #one 2* for imaginary part of fourier transform, one +1 for domain_dim
        if self.patch:
            self.patch_dim = (self.domain_dim+2)*(self.patch_size**2) if self.fourier else (self.domain_dim+1)*(self.patch_size**2)

        self.im_size = im_size

        self.set_positional_encoding()


        encoder_layer = TransformerEncoderLayer_Conv_E(
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

        if self.patch:

            ###
            self.linear_in = nn.Linear(3,d_model)
            ###
            #channel in, channel out, kernel size, padding = kernel size // 2 for kernel size odd
            if self.fourier:
                self.size_row = self.im_size
                self.size_col = self.im_size//2
                #self.layer_norm = nn.LayerNorm((2, self.im_size, self.im_size//2))
                self.linear_out = nn.Linear(d_model, 2*self.patch_size**2)
                #self.conv_out = nn.Conv2d(1,2,1, padding=0)
            else: 
                self.size_row = self.im_size
                self.size_col = self.im_size
                self.linear_out = nn.Linear(d_model,1)
            self.num_patches = (self.size_row*self.size_col)//(self.patch_size**2)

        else:
            if append_position_to_x:
                #linear layer to transform the input to the right dimension if positions are appended to input
                # (batch_size, seq_len, input_dim+domain_dim)
                self.linear_in = nn.Linear(input_dim + domain_dim, d_model)
            else:
                self.linear_in = nn.Linear(input_dim, d_model)
    
            self.linear_out = nn.Linear(d_model, output_dim)
            

    def set_im_size(self, new_im_size,new_patch_size):
        # Update the im_size attribute
        self.patch_size = new_patch_size
        self.im_size = new_im_size
        self.size_row = new_im_size
        self.size_col = new_im_size //2 if self.fourier else new_im_size
        self.num_patches = (self.size_row*self.size_col)//(self.patch_size**2)

        for encoder_layer in self.encoder.layers:
            encoder_layer.self_attn.scaled_dot_product_attention.scale[0]=torch.tensor(new_im_size**2)
            #encoder_layer.self_attn.im_size=new_im_size

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

        # for continuous time positional encoding (assumes square spatial domain)
        even_inds = torch.arange(0, self.d_model, 2).unsqueeze(0)
        odd_inds = torch.arange(1, self.d_model, 2).unsqueeze(0)
        self.register_buffer('even_inds', even_inds)
        self.register_buffer('odd_inds', odd_inds)

    def pe_continuous(self, coords):
        '''generate the positional encoding for coords'''
        # .to() sends the tensor to the device of the argument
        pe = torch.zeros(coords.shape[0], self.d_model).to(coords)
        pe[:, 0::2] = torch.sin(10**self.pos_enc_coeff * coords[:,0] * 10**(-4 * self.even_inds / self.d_model))
        pe[:, 1::2] = torch.cos(10**self.pos_enc_coeff * coords[:,0] * 10**(-4 * self.odd_inds / self.d_model))
        for i in range(1, self.domain_dim):
            pe[:, 0::2] = pe[:, 0::2] * torch.sin(10**self.pos_enc_coeff * coords[:,i] * 10**(-4 * self.even_inds / self.d_model))
            pe[:, 1::2] = pe[:, 1::2] * torch.cos(10**self.pos_enc_coeff * coords[:,i] * 10**(-4 * self.odd_inds / self.d_model))
        return pe

    def positional_encoding(self, x, coords):
        # x: (batch_size, seq_len, input_dim)
        # pe: (1, seq_len, d_model)
        # x + pe[:, :x.size(1)]  # (batch_size, seq_len, d_model)
        if self.use_positional_encoding=='discrete':
            pe = self.pe_discrete[:, :x.size(1)]
        elif self.use_positional_encoding=='continuous':
            pe = self.pe_continuous(coords)
        else: # no positional encoding
            # .to() sends the tensor to the device of the argument
            pe = torch.zeros(x.shape).to(x)

        return pe

    def apply_positional_encoding(self, x, coords):
        pe = self.positional_encoding(x, coords)
        if self.include_y0_input:
            x[:, self.output_dim:, :] += pe
            #'include_y0_input': ['uniform', 'staggered', False],
            if self.include_y0_input == 'uniform':
                x[:, :self.output_dim, :] += torch.tensor(2).to(x)
            elif self.include_y0_input == 'staggered':
                x[:, :self.output_dim, :] += torch.arange(2, self.output_dim+2).unsqueeze(0).unsqueeze(2).to(x)
            else:
                raise ValueError('include_y0_input must be one of [uniform, staggered, False]')
        else:
            x += pe
        return x
    
    def chop_to_patches(self,x):

 
        num_channels = x.shape[1]
        # - shape is now (batch_size, num_channels, num_patch_per_row, patch_size, num_patch_per_col, patch_size)
        chopped_x = x.reshape(x.shape[0], num_channels, self.size_row // self.patch_size, self.patch_size,
                      self.size_col // self.patch_size, self.patch_size)
        
        # - shape is now (batch_size, num_patch_per_row, num_patch_per_col, num_channels, patch_size, patch_size)
        chopped_x = torch.permute(chopped_x, (0,2,4,1,3,5))

        ###########
        ###test####
        ###########

        chopped_x = torch.permute(chopped_x, (0,3,4,5,1,2))
        # - shape is now (batch_size, num_channels, patch_size, patch_size, num_patch_per_row, num_patch_per_col)
        chopped_x = chopped_x.reshape(chopped_x.shape[0], chopped_x.shape[1], chopped_x.shape[2], chopped_x.shape[3], chopped_x.shape[4]*chopped_x.shape[5])
        # - shape is now (batch_size, num_channels, patch_size, patch_size, num_patches)
        chopped_x = torch.permute(chopped_x, (0,4,2,3,1))
        # - shape is now (batch_size, num_patches, patch_size, patch_size, num_channels)

        ###########
        ###########

        return chopped_x
    
    def posemb_sincos_2d(self, h, w, dim, temperature= 10000.0, dtype = torch.float32):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature ** omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return pe.type(dtype)


    def forward(self, x, y=None, coords_x=None, coords_y=None, x_train_fourier_normalizer=None):

        if self.patch:
            residual = x.to(torch.float32)
            x = torch.unsqueeze(x, 1) # (batch_size, 1, rows, cols)
            coords_x = coords_x.squeeze(2) #(domain_dim, rows, cols)
            coords_x = coords_x.repeat(x.shape[0],1,1,1) # (batch_size, domain_dim, rows, cols)
            if self.fourier:
                x = self.fourier_transformation(x, x_train_fourier_normalizer) # (batch_size,2, rows, cols//2+1)
                #x = self.layer_norm(x)
                coords_x = coords_x[...,:self.size_col]
            #addition of coordinates to input as channels
            x = torch.cat((x,coords_x), dim=1) # (batch_size, 1 or 2+domain_dim), rows, cols)
            x = self.chop_to_patches(x)
            ##############
            ###test###### uncomment
            #now flatten the patches
            '''
            x = x.reshape(x.shape[0], self.num_patches, self.patch_dim).to(torch.float32) # (batch_size, num_patches, patch_size)
            '''
            x = x.to(torch.float32)
            ###############

        if self.include_y0_input:
            #this only works when the input dimension is 1, indeed how would you concatenate initial condition with the input otherwise?
            # x = x.permute(1,0,2) # (seq_len, batch_size, dim_state)
            # reshape to make sure we have dim (batch,1,dim_output)
            initial_cond = y[:, 0:1, :].permute(0, 2, 1)  # (batch_size, dim_state, 1)
            # (batch_size, seq_len+output_dim, dim_state)
            x = torch.cat((initial_cond, x), dim=1)

        if self.append_position_to_x:
            append = coords_x.permute(2,0,1).repeat(x.shape[0],1,1)
            x = torch.cat((x, append), dim=2)

        x = self.linear_in(x)  

    
        # can use first time because currently all batches share the same time discretization
        x = self.apply_positional_encoding(x, coords_x) # coords_x is "time" for 1D case


        if self.use_transformer:
            x = self.encoder(x)  # (batch_size, seq_len, dim_state)
    
            
        #check the reshaping!!!
        x = x.reshape(x.shape[0],self.im_size//self.patch_size, self.im_size//self.patch_size ,self.patch_size,self.patch_size, self.d_model)
        x = x.permute(0,1,3,2,4,5)
        x = x.reshape(x.shape[0],self.size_row,self.size_col,self.d_model)
        

        x = self.linear_out(x)  # (batch_size, seq_len, output_dim)? I think it's (batch_size, seq_len, output_dim)
        x = x.squeeze(3)

        

        #learning residual
        #x = x + residual

        


        if self.include_y0_input:
            return x[:, self.output_dim:, :]
        else:
            return x
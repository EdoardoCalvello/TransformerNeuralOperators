import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        #d_model* or just d_model?
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([d_k])), requires_grad=False)
        self.dropout = nn.Dropout(dropout)

    def custom_softmax(self, x, coords, dim=-1):

        exp_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
        #reweighting of exp_x along seq_len dimension
        softmax_x = exp_x / (coords*exp_x[...,1:]).sum(dim=dim, keepdim=True)

        return softmax_x

    def forward(self, query, key, value, coords, key_padding_mask=None):
        # Custom logic for attention calculation
        
        scores = torch.einsum("bhld,bhsd->bhls", query, key) / self.scale

        #makes sure that if domain_dim is not 1, then coords is handled differently
        if coords.shape[1]==1:
            coords = torch.abs(coords[1:,...] - coords[:-1,...])
            coords = coords.permute(1,2,0).unsqueeze(0)
            #coords is now a vector of distances between coordinates such that the shape is broadcastable with scores[...,1:] (1,1,1,seq_len-1)
        else:
            coords = 1

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attention_weights = self.custom_softmax(scores, coords, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        #reweighting of value along seq_len dimension
        value = coords.permute(0,1,3,2)*value[...,1:,:]
        output = torch.einsum("bhls,bhsd->bhld", attention_weights[...,1:], value)
       
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.W_q = nn.Linear(d_model, nhead*self.d_k)
        self.W_k = nn.Linear(d_model, nhead*self.d_k)
        self.W_v = nn.Linear(d_model, nhead*self.d_k)
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_k, dropout=dropout)
        self.W_o = nn.Linear(nhead*self.d_k, d_model)

    def split_heads(self, x):
        #batch_size, seq_length, d_model = x.size()
        batch_size = x.shape[0]
        return x.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        #batch_size, nhead, seq_length, d_k = x.size()
        batch_size = x.shape[0]
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)
        
    def forward(self, x, coords_x, mask=None):

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, coords_x, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, activation):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.activation = getattr(F, activation)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, activation="relu", norm_first=True, do_layer_norm=True, dim_feedforward=2048, batch_first=True):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, dim_feedforward, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, coords_x, mask=None):
        attn_output = self.self_attn(x, coords_x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, x, coords_x, mask=None):
        for layer in self.layers:
            x = layer(x, coords_x, mask=mask)
        return x

###############################################################################################################
###############################################################################################################
#MODULES FOR OPERATOR ARCHITECTURE
###############################################################################################################
###############################################################################################################
    




class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, nhead):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.nhead = nhead

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, num_patches, in_channel, x,y ), (in_channel, out_channel, x,y, nhead) -> (batch, num_patches, out_channel, x,y, nhead)
        return torch.einsum("bnixy,ioxyh->bnoxyh", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        num_patches = x.shape[1]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x = torch.permute(x, (0,1,4,2,3))
        #x is of shape (batch, num_patches, d_model, x, y)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, num_patches, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, self.nhead, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :, :self.modes1, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, :, -self.modes1:, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-3, -2))
        x = torch.permute(x, (0,1,3,4,2,5))
        #x is of shape (batch, num_patches, x, y, d_model, nhead)

        return x


class MultiheadAttention_Operator(nn.Module):
    def __init__(self, d_model, nhead, modes1, modes2, im_size, dropout=0.1):
        super(MultiheadAttention_Operator, self).__init__()
        self.nhead = nhead

        self.query_operator = SpectralConv2d(d_model, d_model, modes1, modes2, nhead)
        self.key_operator = SpectralConv2d(d_model, d_model, modes1, modes2, nhead)
        self.value_operator = SpectralConv2d(d_model, d_model, modes1, modes2, nhead)

        self.scaled_dot_product_attention = ScaledDotProductAttention_Operator(d_model, im_size, dropout=dropout)

        self.out_linear = nn.Linear(d_model*nhead, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):

        batch, num_patches, patch_size, patch_size, d_model = x.size()
        query = self.query_operator(x).permute(0,5,1,2,3,4)
        key = self.key_operator(x).permute(0,5,1,2,3,4)
        value = self.value_operator(x).permute(0,5,1,2,3,4)
        # query, key, and value are of shape (batch, nhead, num_patches,x,y , d_model)

        # Scaled Dot Product Attention
        attention_output, _ = self.scaled_dot_product_attention(query, key, value, key_padding_mask=key_padding_mask)

        # Reshape and linear transformation
        attention_output = attention_output.reshape(batch, num_patches, patch_size, patch_size,-1)
        output = self.out_linear(attention_output)
        output = self.dropout(output)

        return output

class ScaledDotProductAttention_Operator(nn.Module):
    def __init__(self, d_model, im_size, dropout=0.1):
        super(ScaledDotProductAttention_Operator, self).__init__()
        #d_model* or just d_model?
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([((im_size)**4)])), requires_grad=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        # Custom logic for attention calculation
        scores = torch.einsum("bnpxyd,bnqxyd->bnpq", query, key) / self.scale
    
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.einsum("bnpq,bnqxyd->bnpxyd", attention_weights, value)
        output = output.permute(0,2,3,4,5,1)

        return output, attention_weights

class TransformerEncoderLayer_Operator(nn.Module):#
    def __init__(self, d_model, nhead, dropout=0.1, activation="relu", norm_first=True, do_layer_norm=True, dim_feedforward=2048, modes=None, patch_size=1, im_size=64, batch_first=True):
        super(TransformerEncoderLayer_Operator, self).__init__()

        # Self-attention layer
        if modes is None:
            modes2 = patch_size//2+1
            modes1 = patch_size//2+1
        else:
            modes1 = modes[0]
            modes2 = modes[1]
        #or im_size?
        self.self_attn = MultiheadAttention_Operator(d_model, nhead, modes1, modes2, im_size, dropout=dropout)

        # Feedforward layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation function
        self.activation = getattr(F, activation)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Whether to normalize before the self-attention layer
        self.norm_first = norm_first

        # Whether to apply layer normalization at the end
        self.do_layer_norm = do_layer_norm

    def forward(self, x, mask=None):
        if self.norm_first:
            x = self.norm1(x)

        # Self-attention
        attn_output = self.self_attn(x, key_padding_mask=mask)

        # Residual connection and layer normalization
        x = x + self.dropout(attn_output)
        if not self.norm_first:
            x = self.norm1(x)

        # Feedforward layer
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))

        # Residual connection and layer normalization
        x = x + self.dropout(ff_output)
        if self.do_layer_norm:
            x = self.norm2(x)

        return x

class TransformerEncoder_Operator(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder_Operator, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
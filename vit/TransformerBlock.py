import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention

class TransformerBlock(nn.Module):
    ''' Implements the Transformer Encoder block:
            [LayerNorm, Multi-head attention, residual,
             LayerNorm, MLP, residual]

    '''
    def __init__(self, input_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # TODO define LayerNorm (use nn.LayerNorm -- https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

        # TODO define MLP (one linear, one activation function, one linear -- go back to the ViT paper to find which
        #      activation function you should be using)
        self.mlp =  nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(mlp_dim, input_dim)
        )

        # TODO define MultiHeadAttention
        self.multihead_attention = MultiHeadAttention(input_dim, num_heads, dropout)

    def forward(self, x):
        # TODO apply LayerNorm
        x1 = self.layer_norm1(x)
        # TODO define Q, K, V
        Q = x1
        K = x1
        V = x1
        # TODO pass (Q, K, V) to MultiHeadAttention
        attention_output, _ = self.multihead_attention(Q,K,V)
        # TODO sum with 1st residual connection
        attention_output = attention_output + x
        # TODO apply LayerNorm
        norm1out = self.layer_norm2(attention_output)
        # TODO pass to MLP
        mlp_out = self.mlp(norm1out)
        # TODO sum with 2nd residual connection
        x = mlp_out + attention_output
        return x

import torch
import numpy as np
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    ''' Implements the Scaled Dot Product Attention function as defined in
        Attention Is All You Need (Vaswani et al., 2017).
        Args:
        - d_k (int): the dimension of the key vectors
    '''
    def __init__(self, d_k, dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q, K, V):
        ''' Computes the scaled dot product attention :
            Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V

            Args:
            - Q (torch.Tensor): the query tensor of shape (batch_size, num_queries, dim_k)
            - K (torch.Tensor): the key tensor of shape (batch_size, num_keys, dim_k)
            - V (torch.Tensor): the value tensor of shape (batch_size, num_keys, dim_v)

            Returns:
            - attention_output (torch.Tensor): the output tensor of shape (batch_size, num_queries, dim_v), obtained
            by applying the attention mechanism to the input values V using the input queries Q and keys K
            - attention_scores (torch.Tensor): the attention tensor of shape (batch_size, num_queries, num_keys),
            representing the attention scores between each query and key (output of softmax)
        '''
        # TODO: implement the scaled dot product attention following the above
        # formula.

        attention = torch.bmm(Q, K.permute(0, 2, 1))
        dimension = torch.as_tensor(self.d_k, dtype=attention.dtype, device=attention.device).sqrt()
        attention = attention/dimension
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        output = torch.bmm(attention, V)
        # HINT: Remember that we might be working with batched data, which will
        # impact the shape of the input vectors. This will be important for how
        # you compute the tranpose of K and the softmax.
        attention_output, attention_scores = output, attention

        return attention_output, attention_scores

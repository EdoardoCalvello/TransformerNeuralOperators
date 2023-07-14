import torch.nn as nn
import torch
from ScaledDotProductAttention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    ''' Implements the Multi-Head Attention block as defined in Attention Is All
        You Need (Vaswani et al., 2017).
        Args:
        - d_input (int): the dimension of the input vectors
        - num_heads (int): the number of attention heads to use
        - dropout (float): the dropout rate to use between layers
    '''
    def __init__(self, d_input, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_input // num_heads
        self.dropout = nn.Dropout(dropout)
        # Note: here we are stacking the W_ matrices across heads to streamline the matrix multiplications
        # of the linear layers for V, K, Q. We will split outputs per head after applying the W_ and before
        # feeding them to the scale dot-product attention blocks. Each linear layer should have an output of
        # dimension d_k, and we have num_heads such parallel layers for V, K, Q. For streamlining, we hence
        # have one output of dimension (num_heads * d_k) which will later be split into num_heads outputs of
        # dimension d_k.
        self.W_Q = nn.Linear(d_input, num_heads * self.d_k)
        self.W_K = nn.Linear(d_input, num_heads * self.d_k)
        self.W_V = nn.Linear(d_input, num_heads * self.d_k)
        # Note: W_O corresponds to the last linear transformation of the MultiHead Attention block
        self.W_O = nn.Linear(num_heads * self.d_k, d_input)

    def forward(self, Q, K, V):
        ''' Computes the multi-head attention mechanism:
            MultiHeadAttention(Q,K,V) = Concat(head_1, head_2, ..., head_h)W_O
            where head_i = Attention(QW_Q_i, KW_K_i, VW_V_i)

            Args:
            - Q (torch.Tensor): the query tensor of shape (batch_size, num_queries, d_input)
            - K (torch.Tensor): the key tensor of shape (batch_size, num_keys, d_input)
            - V (torch.Tensor): the value tensor of shape (batch_size, num_keys, d_input)

            Returns:
            - output (torch.Tensor): the output tensor of shape (batch_size, num_queries, d_input), obtained
              by applying the multi-head attention mechanism to the input values V using the input queries Q and keys K
            - attention (torch.Tensor): the attention tensor of shape (batch_size, num_heads, num_queries, num_keys),
              representing the attention scores between each query and key for each head
        '''
        batch_size = Q.size(0)

        # Apply streamlined W_ linear transformations stacked across heads, then reshape
        # to split into num_heads outputs of dimension d_k.
        # TODO: why do we need the "transpose(1,2)" here?
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # TODO: here call your implementation of ScaledDotProductAttention or PyTorch's version
        #attention = torch.nn.functional.scaled_dot_product_attention(self.d_k)
        attention_output = torch.nn.functional.scaled_dot_product_attention(Q,K,V)

        # Note: the attention_output tensor has shape (batch_size, num_heads, num_queries, d_k)
        # In order to combine the outputs from the different attention heads, we need to concatenate
        # them along the feature dimension, which is d_k.
        # 1. swap the num_heads and num_queries dimensions
        # 2. create a new tensor that is laid out in memory in a contiguous manner
        # 3. reshape the tensor into (batch_size, num_queries, num_heads * d_k)
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.contiguous()
        attention_output = attention_output.view(batch_size, -1, self.num_heads * self.d_k)

        # TODO: fill in the code below with any missing operation to complete the block
        #attention_output = torch.mean(attention_output, dim=2)
        attention_output = self.W_O(attention_output)
        attention_output = self.dropout(attention_output)

        return attention_output, None

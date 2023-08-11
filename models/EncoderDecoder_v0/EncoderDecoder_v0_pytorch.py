import os
import torch
import torch.nn as nn

# installed pytorch
# from torch.nn import TransformerEncoder
# from torch.nn import TransformerEncoderLayer

# source code
from models.transformer_custom import PositionalEncoding, EncoderLayer, DecoderLayer 

# Define the neural network model
class EncoderDecoder_v0(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1, d_model=32, nhead=8, num_layers=6, dim_feedforward=2048, max_sequence_length=100, dropout=0.1,learning_rate=0.01):
        super(EncoderDecoder_v0, self).__init__()

        self.max_sequence_length = max_sequence_length
        #using linear for embedding
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.decoder_embedding = nn.Linear(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, self.max_sequence_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

        self.learning_rate = learning_rate
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1)
        tgt_mask = (tgt != 0).unsqueeze(1)
        seq_length = src.size(1)
        nopeak_mask = torch.tril(torch.ones(seq_length, seq_length)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        #import pdb; pdb.set_trace()
        return src_mask, tgt_mask


    def forward(self, src, tgt=None, validation=False):
        src_mask, tgt_mask = self.generate_mask(src, tgt) if tgt is not None else (None, None)

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded

        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, None)  # src_mask

        if validation==False:
            tgt_embedded = (self.positional_encoding(self.decoder_embedding(tgt)))#no dropout?
            dec_output = tgt_embedded

            
            #for dec_layer in self.decoder_layers:
            #    dec_output = dec_layer(dec_output, enc_output, None, tgt_mask)  # src_mask, tgt_mask
            
            
            ###delete below####
            output_seq = dec_output[:,0,:].unsqueeze(dim=1)  # Store the first prediction (dummy_input)
            #dec_output = output_seq #delete if you want decoder to learn from unrolling truth
            for i in range(self.max_sequence_length - 1):
                dec_output = tgt_embedded[:,:i+1,:]#.unsqueeze(dim=1) #undelete if you want decoder to learn from unrolling truth
                for dec_layer in self.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, None, None)
                #output_seq = torch.cat((output_seq, dec_output[:,-1,:].unsqueeze(dim=1)), dim=1)
                output_seq = torch.cat((output_seq, output_seq[:,-1,:].unsqueeze(dim=1)+ 0.01*dec_output[:, -1, :].unsqueeze(dim=1)), dim=1)  # Shape: (batch_size, seq_length, d_model)
                #dec_output = output_seq #delete if you want decoder to learn from unrolling truth 
            output = self.fc(output_seq)
            ###delete above####
            
            #output = self.fc(dec_output)
            return output

        if validation==True:
            # Autoregressive decoding during validation and testing
            
            # Start with a dummy token as the initial input to the decoder
            if tgt is not None:
                tgt_embedded = (self.positional_encoding(self.decoder_embedding(tgt))) #no dropout?
                dummy_input = tgt_embedded[:, :1, :]  # Shape: (batch_size, 1, d_model) #src_embedded[:, :1, :]?
            else:
                dummy_input = torch.zeros_like(src_embedded[:, :1, :])  # Shape: (batch_size, 1, d_model) #src_embedded[:, :1, :]? 

            dec_output = dummy_input
            output_seq = dec_output  # Store the first prediction (dummy_input)
            
            for _ in range(self.max_sequence_length - 1):
                for dec_layer in self.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, None, None)
                #output_seq = torch.cat((output_seq, dec_output[:,-1,:].unsqueeze(dim=1)), dim=1)
                output_seq = torch.cat((output_seq, output_seq[:,-1,:].unsqueeze(dim=1)+ 0.01*dec_output[:, -1, :].unsqueeze(dim=1)), dim=1)  # Shape: (batch_size, seq_length, d_model)
                dec_output = output_seq  
            
            output = self.fc(output_seq)

            return output
import torch
import numpy as np
import torch.nn as nn 
from genesis_ILDP.model.common.mask_gen import *

class Feedforward(nn.Module):
    # input dim (batch, sequence_len, d_model)
    def __init__(self, d_model, k, dropout=0.1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_model, k * d_model),
            # nn.ReLU(),
            nn.Mish(),
            nn.Dropout(dropout),  # Dropout after activation
            nn.Linear(k * d_model, d_model)
        )
    def forward(self, x):
        # multiply over the last d_model dim
        return self.layer(x)
    
class MultiHeadAttentionBlock(nn.Module):
    # param: d_model len query, key, value as input, mask, src, src_mask
    def __init__(self, d_model, heads, src = False, ffn_depth = 4, dropout=0.1):
        super().__init__()
        self.ffn = Feedforward(d_model, ffn_depth, dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.norm_self_attn = None
        self.norm_src_attn = None
        self.dropout = nn.Dropout(dropout)  # Dropout for residual connections
        if not src:
            self.norm_self_attn = nn.LayerNorm(d_model) # normalize across the embedding dim
            # independentely!
            self.self_attn = _attn(d_model = d_model, heads = heads, dropout=dropout)
        else:
            self.norm_src_attn = nn.LayerNorm(d_model)
            self.src_attn = _attn(d_model = d_model, heads = heads, dropout=dropout)
        self.heads = heads

    def forward(self, x, x_mask, src = None, src_mask = None):
        if src is None and self.norm_self_attn is not None:

            skip_attn = x
            x = self.norm_self_attn(x)
            if x_mask is None:
                raise ValueError("mask for self attn is empty, pls check the mask for self-attn")
            x = self.self_attn(query = x, key = x, value = x, mask = x_mask)
            x = self.dropout(x) + skip_attn  # Apply dropout before residual

            skip_ffn = x
            x = self.norm_ffn(x)
            x = self.ffn(x)
            x = self.dropout(x) + skip_ffn  # Apply dropout before residual

        elif src is not None and self.norm_src_attn is not None:
            skip_attn = x
            if src_mask is None:
                raise ValueError("src_mask for self attn is empty, pls check the src_mask for attn")
            x = self.norm_src_attn(x)
            x = self.src_attn(query = x, key = src , value = src, mask = src_mask)
            x = self.dropout(x) + skip_attn  # Apply dropout before residual

            skip_ffn = x
            x = self.norm_ffn(x)
            x = self.ffn(x)
            x = self.dropout(x) + skip_ffn  # Apply dropout before residual

        else:
            raise TypeError("Type of MultiHeadAttention does not match the data flow!")

        return x
 
class _attn(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # the input x is in the shape of (Batch_size, seq_len, d_model)
        # TODO check out which way for the initializing these matrices is the best approach
        self.Q = nn.Parameter(torch.randn(size = (d_model, d_model)) * (d_model ** -0.5))
        self.K = nn.Parameter(torch.randn(size = (d_model, d_model)) * (d_model ** -0.5))
        self.V = nn.Parameter(torch.randn(size = (d_model, d_model)) * (d_model ** -0.5))
        self.heads = heads
        self.softmax = nn.Softmax(dim = -1)
        self.attn_dropout = nn.Dropout(dropout)  # Dropout on attention weights
    
    def forward(self, query, key, value, mask):
        # dim (batch_size, seq_len * d_model)
        
        query_sz = query.shape
        key_sz = key.shape
        value_sz = value.shape
        batch_size = query_sz[0]
        
        emb_dim = query_sz[-1]
        assert self.d_model == emb_dim and emb_dim == key_sz[-1] and emb_dim == value_sz[-1]
        # assert mask.shape == (seq_len * seq_len)         
        # we use the additive mask here which assign the -inf to the result we want to mask
        assert query_sz[0] == key_sz[0] and query_sz[0] == value_sz[0]
        assert key_sz[1] == value_sz[1] # ensure that seq_len for key and value is the same
        d_heads = self.d_model // self.heads
        Q_x = (query @ self.Q).reshape(query_sz[0], query_sz[1], self.heads, -1)
        K_x = (key @ self.K).reshape(key_sz[0], key_sz[1], self.heads, -1)
        V_x = (value @  self.V).reshape(value_sz[0], value_sz[1], self.heads, -1) 
        
        Q_x = torch.transpose(Q_x, dim0=1, dim1=2) # change into (batch, heads, seq_len, d_heads)
        K_x = torch.transpose(K_x, dim0=1, dim1=2)
        V_x = torch.transpose(V_x, dim0=1, dim1=2)

        Cor_x = torch.matmul(Q_x, torch.transpose(K_x, dim0=-2, dim1=-1)) # output (batch, heads, seq_len_query, seq_len_key)
        assert mask.shape[0] == query_sz[1] and mask.shape[1] == key_sz[1] # ensure the mask fits the Cor_qk shape
        mask = mask.reshape(1, 1, *(mask.shape))

        Cor_masked = Cor_x + mask

        scale = torch.sqrt(torch.tensor(d_heads, device = query.device, dtype=torch.float32))
        attn_weights = self.softmax(Cor_masked / scale)
        attn_weights = self.attn_dropout(attn_weights)  # Apply dropout on attention weights
        attn_raw = attn_weights @ V_x # output shape (batch_size, n_heads, seq_len, d_heads)
        attn = torch.transpose(attn_raw, dim0=1, dim1=2).reshape(batch_size, -1, self.d_model)
        
        return attn 
        
    
class Encoder(nn.Module):
    # the input for the encoder should be the shape exactly equals to the obs and other cond information used for the decoder
    def __init__(self, n_layers, d_model = 512, heads = 6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.MultiheadAttention = nn.ModuleList([MultiHeadAttentionBlock(d_model, heads, dropout=dropout) for _ in range(self.n_layers)])
        # self.embedding = nn.Linear()
    def forward(self, x, x_mask):
        # the mask sz: (seq_len, seq_len)
        for i in range(self.n_layers):
            x = self.MultiheadAttention[i](x = x, x_mask = x_mask, src = None, src_mask = None)
        return x
    
class Decoder(nn.Module):
    # input param: memory(batch_size, seq_len_memory, d_model), x(batch_size, seq_len_memory, d_model)
    def __init__(self, n_layers, d_model=512, heads = 6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.MultiHeadSelfAttention = nn.ModuleList([MultiHeadAttentionBlock(d_model, heads, src=False, dropout=dropout) for _ in range(self.n_layers)])
        self.MultiHeadCrossAttention = nn.ModuleList([MultiHeadAttentionBlock(d_model, heads, src=True, dropout=dropout) for _ in range(self.n_layers)])

    def forward(self, x, x_mask, src, src_mask):
        for i in range(self.n_layers):
            x = self.MultiHeadSelfAttention[i](x = x, x_mask = x_mask, src = None, src_mask = None)
            x = self.MultiHeadCrossAttention[i](x = x, x_mask = x_mask, src = src, src_mask = src_mask)
        return x
    

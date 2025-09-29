import torch
import numpy as np
import torch.nn as nn 
from genesis_ILDP.model.common.mask_gen import *

class Feedforward(nn.Module):
    # input dim (batch, sequence_len, d_model)
    def __init__(self, d_model, k):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_model, k * d_model), 
            # nn.ReLU(),
            nn.Mish(), 
            nn.Linear(k * d_model, d_model)
        )
    def forward(self, x):
        # maltiply over the last d_model dim 
        return self.layer(x)
    
class MultiHeadAttentionBlock(nn.Module):
    # param: d_model len query, key, value as input, mask, src, src_mask 
    def __init__(self, d_model, heads, src = False, ffn_depth = 4):
        super().__init__()
        self.ffn = Feedforward(d_model, ffn_depth)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.norm_self_attn = None
        self.norm_src_attn = None
        if not src:
            self.norm_self_attn = nn.LayerNorm(d_model) # normalize across the embedding dim 
            # independentely!
            self.self_attn = _attn(d_model = d_model, heads = heads)
        else:
            self.norm_src_attn = nn.LayerNorm(d_model)
            self.src_attn = _attn(d_model = d_model, heads = heads)
        self.heads = heads

    def forward(self, x, x_mask, src = None, src_mask = None):
        if src is None and self.norm_self_attn is not None:

            skip_attn = x
            x = self.norm_self_attn(x)
            if x_mask is None: 
                raise ValueError("mask for self attn is empty, pls check the mask for self-attn")
            x = self.self_attn(query = x, key = x, value = x, mask = x_mask)
            x = x + skip_attn

            skip_ffn = x 
            x = self.norm_ffn(x)
            x = self.ffn(x) + skip_ffn

        elif src is not None and self.norm_src_attn is not None:
            skip_attn = x
            if src_mask is None: 
                raise ValueError("src_mask for self attn is empty, pls check the src_mask for attn")
            x = self.norm_src_attn(x)
            x = self.src_attn(query = x, key = src , value = src, mask = src_mask)
            x = x + skip_attn

            skip_ffn = x 
            x = self.norm_ffn(x)
            x = self.ffn(x) + skip_ffn

        else: 
            raise TypeError("Type of MultiHeadAttention does not match the data flow!")

        return x
 
class _attn(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        # the input x is in the shape of (Batch_size, seq_len, d_model)
        # TODO check out which way for the initializing these matrices is the best approach 
        self.Q = nn.Parameter(torch.randn(size = (d_model, d_model)) * (d_model ** -0.5))
        self.K = nn.Parameter(torch.randn(size = (d_model, d_model)) * (d_model ** -0.5))
        self.V = nn.Parameter(torch.randn(size = (d_model, d_model)) * (d_model ** -0.5))
        self.heads = heads
        self.softmax = nn.Softmax(dim = -1)
    
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

        scale = torch.sqrt(torch.tensor(self.heads, device = query.device, dtype=torch.float32))
        attn_raw = self.softmax(Cor_masked / scale) @ V_x # output shape (batch_size, n_heads, seq_len, d_heads)
        attn = torch.transpose(attn_raw, dim0=1, dim1=2).reshape(batch_size, -1, self.d_model)
        
        return attn 
        
    
class Encoder(nn.Module):
    # the input for the encoder should be the shape exactly equals to the obs and other cond information used for the decoder 
    def __init__(self, n_layers, d_model = 512, heads = 6):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.MultiheadAttention = nn.ModuleList([MultiHeadAttentionBlock(d_model, heads) for _ in range(self.n_layers)])
        # self.embedding = nn.Linear()
    def forward(self, x, x_mask):
        # the mask sz: (seq_len, seq_len)
        for i in range(self.n_layers):
            x = self.MultiheadAttention[i](x = x, x_mask = x_mask, src = None, src_mask = None)
        return x
    
class Decoder(nn.Module):
    # input param: memory(batch_size, seq_len_memory, d_model), x(batch_size, seq_len_memory, d_model)
    def __init__(self, n_layers, d_model=512, heads = 6):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers 

        self.MultiHeadSelfAttention = nn.ModuleList([MultiHeadAttentionBlock(d_model, heads, src=False) for _ in range(self.n_layers)])
        self.MultiHeadCrossAttention = nn.ModuleList([MultiHeadAttentionBlock(d_model, heads, src=True) for _ in range(self.n_layers)])

    def forward(self, x, x_mask, src, src_mask):
        for i in range(self.n_layers):
            x = self.MultiHeadSelfAttention[i](x = x, x_mask = x_mask, src = None, src_mask = None)
            x = self.MultiHeadCrossAttention[i](x = x, x_mask = x_mask, src = src, src_mask = src_mask)        
        return x
    
# TODO add dropout for the transformer layers value passing 
class Encoder_Decoder(nn.Module):
    def __init__(self, n_layers_encoder, n_layers_decoder, 
                 enc_in_dim, dec_in_dim, dec_out_horizon, 
                 d_model = 512, heads = 6, is_causal = True):
        super().__init__()
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder 
        self.d_model = d_model
        self.heads = heads  
        self.is_causal = is_causal
        self.SOS = nn.Parameter(torch.randn(size=(self.d_model, ))) # the start of sentence embedding learned varaible 
        # Or the upper could be change into other visual information that also work as cond ?

        self.enc_in_dim = enc_in_dim
        self.dec_in_dim = dec_in_dim
        self.horizon = dec_out_horizon 

        self.embedding_enc = nn.Linear(self.enc_in_dim, self.d_model)
        self.embedding_dec = nn.Linear(self.dec_in_dim, self.d_model)

        self.inverse_embedding_dec = nn.Linear(self.d_model, self.dec_in_dim)

        self.encoder = Encoder(self.n_layers_encoder, self.d_model, self.heads)
        self.decoder = Decoder(self.n_layers_decoder, self.d_model, self.heads)

        self.causal_mask = torch.tril(torch.ones(self.horizon, self.horizon))

    def _emb(self, x, is_enc):
        if is_enc:
            return self.embedding_enc(x)
        else:
            return self.embedding_dec(x)
        
    def _inv_emb(self, x, ):
        """
        Param:
            x: (batch_size, seq_len, d_model)
        Return:
            inv_x: (batch_size, seq_len, dec_emb)
        """
        return self.inverse_embedding_dec(x)
        
    def _get_src(self, cond, cond_mask=None):
        src = self.encoder(cond, cond_mask)
        return src;
    
    def predict_onetime_seq(self, x, cond, x_mask=None, cond_mask=None):
        """
        predict_onetime_seq is special designed for training. 

        Param: x: the answer to the prediction seq which is right moved for predicting the next token
               cond: the cond used for encoding src and help prediction 
               x_mask: [None] It should be None since we use causal mask for almost any case. A default causal mask is applied.
               cond_mask: [(batch_sz, cond_len, cond_len)] 
        """

        if x_mask is not None: 
            raise ValueError("Only support training using causal mask!")
        
        if x.shape[1] != self.horizon:
            raise ValueError("Training input inconsisent with expected output length")
        
        if x.shape[2] != self.dec_in_dim:
            raise ValueError("The in_dim from the decoder does not match the embedding for decoder shape")
        
        x = self._emb(x, is_enc=False) # embed the decoder input
        cond = self._emb(cond, is_enc=True) # embed the encoder input

        memory = self.encoder(cond, cond_mask)
        x_input = torch.concatenate([
            self.SOS, 
            x[:, :-1, :] # discard the last value and add the initial symbol for predicting 
        ], dim=1)
        output_emb = self._predict_full_seq(x_input, src=memory, x_mask=None, src_mask=cond_mask)

        output = self._inv_emb(output_emb)
        return output

    def _predict_full_seq(self, x, src, x_mask=None, src_mask=None):
        # the x mask should be the causal mask if enable the causal mask, 
        # or if the mask is None and not in causal mode, the mask should be dummy
        x_len = x.shape[-2] # used to generate the correct mask shape
        if x_mask is None and self.is_causal:
            x_mask = self.causal_mask[:x_len, :x_len]
        elif x_mask is None:
            x_mask = DummyMask(width=self.horizon, height=self.horizon)
        else:
            assert x_mask.shape == (self.horizon, self.horizon)
        
        output = self.decoder(x = x, x_mask=x_mask, src=src, src_mask=src_mask)
        return output
    
    def _predict_last_token(self, x, src, x_mask=None, src_mask=None):
        output = self._predict_seq(x, src=src, x_mask=x_mask, src_mask=src_mask)

        # return the last token which is our prediction for the next token in embedded dim 
        return output[:, -1, :] 
    
    def gen_seq(self, cond, x_mask=None, cond_mask=None, src_mask=None):
        """
        gen_seq is used for forward pass for predicting a full action sequence from cond, and used cond_mask, x_mask, src_mask
        """
        memory = self.encoder(cond, cond_mask)
        batch_size = memory.shape[0]
        x = self.SOS.repeat(batch_size, 1, 1)
        memory_mask = src_mask 

        for _ in range(self.horizon):
            token_new = self._predict_last_token(x=x, src=memory, x_mask = None, src_mask=memory_mask) # Use the default causal mask 
            x = torch.concatenate(
                (x, token_new), 
                dim=1
            )
        output_raw = x[:, 1:, :] # remove the first sos from the action sequence

        output = self.inverse_embedding_dec(output_raw)
        return output
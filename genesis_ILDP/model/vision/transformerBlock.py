import torch
import torch.nn as nn
import numpy as np
from genesis_ILDP.model.transformer import *

class Encoder_Decoder(nn.Module):
    def __init__(self, n_layers_encoder, n_layers_decoder,
                 enc_in_dim, dec_in_dim, dec_out_horizon,
                 d_model = 512, heads = 6, is_causal = True, dropout=0.1):
        super().__init__()
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.d_model = d_model
        self.heads = heads
        self.is_causal = is_causal
        self.enc_in_dim = enc_in_dim
        self.dec_in_dim = dec_in_dim
        self.horizon = dec_out_horizon

        self.SOS = nn.Parameter(torch.randn(size=(self.d_model, )))

        self.embedding_enc = nn.Sequential(
            nn.Linear(self.enc_in_dim, self.d_model),
            nn.Dropout(dropout)
        )
        self.embedding_dec = nn.Sequential(
            nn.Linear(self.dec_in_dim, self.d_model),
            nn.Dropout(dropout)
        )
        self.output_projection = nn.Linear(self.d_model, self.dec_in_dim)

        self.encoder = Encoder(self.n_layers_encoder, self.d_model, self.heads, dropout)
        self.decoder = Decoder(self.n_layers_decoder, self.d_model, self.heads, dropout)

        causal_mask = torch.tril(torch.ones(self.horizon, self.horizon))
        causal_mask = torch.where(causal_mask == 1, 0.0, float('-inf'))
        self.register_buffer('causal_mask', causal_mask)

    def _embed_encoder_input(self, x):
        return self.embedding_enc(x)

    def _embed_decoder_input(self, x):
        return self.embedding_dec(x)

    def _project_to_output(self, x):
        return self.output_projection(x)
    
    def forward(self, x, cond, x_mask=None, cond_mask=None, src_mask=None):
        """
        Forward pass for training with teacher forcing.

        Args:
            x: ground truth action sequence (batch, horizon, dec_in_dim)
               Will be right-shifted for teacher forcing
            cond: conditioning input (batch, cond_len, enc_in_dim) - raw, will be embedded
            x_mask: Should be None - causal mask is applied automatically
            cond_mask: mask for encoder self-attention (cond_len, cond_len)
            src_mask: mask for decoder cross-attention to encoder memory (dec_len, cond_len) or None

        Returns:
            output: predicted action sequence (batch, horizon, dec_in_dim)
        """

        if x_mask is not None:
            raise ValueError("Only support training using causal mask!")

        if x.shape[1] != self.horizon:
            raise ValueError("Training input inconsisent with expected output length")

        if x.shape[2] != self.dec_in_dim:
            raise ValueError("The in_dim from the decoder does not match the embedding for decoder shape")

        x_emb = self._embed_decoder_input(x)
        cond_emb = self._embed_encoder_input(cond)

        memory = self.encoder(cond_emb, cond_mask)

        x_input = torch.concatenate([
            self.SOS.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, -1),
            x_emb[:, :-1, :]
        ], dim=1)

        output_emb = self._decode_full_seq(x_input, src=memory, x_mask=None, src_mask=src_mask)
        output = self._project_to_output(output_emb)
        return output

    def _decode_full_seq(self, x, src, x_mask=None, src_mask=None):
        x_len = x.shape[-2]
        if x_mask is None and self.is_causal:
            x_mask = self.causal_mask[:x_len, :x_len]
        elif x_mask is None:
            x_mask = DummyMask(width=x_len, height=x_len)
        else:
            assert x_mask.shape == (x_len, x_len)

        return self.decoder(x=x, x_mask=x_mask, src=src, src_mask=src_mask)

    def _decode_next_token(self, x, src, x_mask=None, src_mask=None):
        output = self._decode_full_seq(x, src=src, x_mask=x_mask, src_mask=src_mask)
        return output[:, -1, :] 
    
    def generate(self, cond, cond_mask=None, src_mask=None):
        """
        Autoregressive generation for inference/evaluation.

        Args:
            cond: conditioning input (batch, cond_len, enc_in_dim)
            cond_mask: mask for encoder self-attention (cond_len, cond_len)
            src_mask: mask for decoder cross-attention (dec_len, cond_len) or None

        Returns:
            output: generated action sequence (batch, horizon, dec_in_dim)
        """
        cond_emb = self._embed_encoder_input(cond)
        memory = self.encoder(cond_emb, cond_mask)
        batch_size = memory.shape[0]

        x = self.SOS.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)

        for _ in range(self.horizon):
            token_new = self._decode_next_token(x=x, src=memory, x_mask=None, src_mask=src_mask)
            x = torch.concatenate((x, token_new), dim=1)

        output_emb = x[:, 1:, :]
        output = self._project_to_output(output_emb)
        return output
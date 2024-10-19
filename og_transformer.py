"""
The original Encoder Decoder transformer architecture. 
Primarily used for language translation where you put your
x through the encoder and your tranlated y_target 
through the decoder  

The decoder autoregressivly generates output tokens conditioned on the
output of the decoder (which uses unidirectional attention)
"""
import torch.nn as nn

from transformer import EncoderBlock, DecoderBlock, PosEncoding

class Encoder(nn.Module):
    def __init__(self, n_layers=6, d_model=512, n_heads=8, dff=2048):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, dff) for _ in range(n_layers)]) 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # apperently adding another Layer norm here is benefitial but
        # we are copying the paper so
        return x 

class Decoder(nn.Module):
    def __init__(self, n_layers=6, d_model=512, n_heads=8, dff=2048):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_heads, dff) for _ in range(n_layers)])
    
    def forward(self, x, cond_attention):
        for layer in self.layers:
            x = layer(x, cond_attention)
        return x

class Transformer(nn.Module):
    def __init__(
            self,
            block_size=512,
            d_model=512,
            in_vocab_s=52000,
            out_vocab_s=52000, 
    ):
        super().__init__()
        self.input_embed = nn.Embedding(in_vocab_s, d_model)
        self.target_embed = nn.Embedding(out_vocab_s, d_model)            
        self.input_pos_enc = PosEncoding(block_size, d_model)
        self.target_pos_enc = PosEncoding(block_size, d_model)
        self.encoder = Encoder()
        self.decoder = Decoder() 
        self.linear = nn.Linear(d_model, out_vocab_s)
    
    def forward(self, x, x_target):
        in_embed = self.input_embed(x)
        in_embed = self.input_pos_enc(in_embed)
        
        tgt_embed = self.target_embed(x_target)
        tgt_embed = self.target_pos_enc(tgt_embed)
        
        cond_attention = self.encoder(in_embed)
        decoded = self.decoder(tgt_embed, cond_attention)

        output = self.linear(decoded)

        return output
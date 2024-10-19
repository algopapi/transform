import torch
import torch.nn as nn

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False):
        super().__init__()
        assert d_model % n_heads == 0
        
        k = d_model // n_heads # number of attention parameters per head

        self.wq_dhk = nn.Parameter(torch.randn(d_model, n_heads, k))
        self.wk_dhk = nn.Parameter(torch.randn(d_model, n_heads, k))
        self.wv_dhk = nn.Parameter(torch.randn(d_model, n_heads, k))
        self.wo_hkd = nn.Parameter(torch.randn(n_heads, k, d_model))
        self.scale = torch.sqrt(torch.tensor(k))
        self.causal = causal 
        
    def forward(self, x_bld, c_bld=None):
        """
        The input comes in as 
        b: batch_size 
        l: sequence length (padded tot he longest sequence)
        d: embedding dimension
        """
        if c_bld is None:
            c_bld = x_bld 
        # Make einstein
        q_blhk = torch.einsum('bld,dhk->blhk', x_bld, self.wq_dhk)
        k_blhk = torch.einsum('bld,dhk->blhk', c_bld, self.wk_dhk)
        v_blhk = torch.einsum('bld,dhk->blhk', c_bld, self.wv_dhk)

        qkt_bhlm = torch.einsum('blhk,bmhk->bhlm', q_blhk, k_blhk) 
        qkt_bhlm = qkt_bhlm / self.scale
        
        if self.causal:
            _, l, _ = x_bld.shape
            mask = torch.triu(torch.ones(l, l), diagonal=1).bool().to(qkt_bhlm.device)
            mask = mask.unsqueeze(0).unsqueeze(0)
            qkt_bhlm = qkt_bhlm.masked_fill(mask, float('-inf'))
            
        # get the weighted values. 
        qkt_bhlm = torch.softmax(qkt_bhlm, dim=-1)
        wtd_values_blhk = torch.einsum('bhlm,bmhk->blhk', qkt_bhlm, v_blhk)
        output_bld = torch.einsum('blhk,hkd->bld', wtd_values_blhk, self.wo_hkd)
        return output_bld

class FFN(nn.Module):
    """ Intuivitively i hav eread somewhere that this model is to capture information gained between the heads? """ 
    def __init__(self, dff, d_model):
        super().__init__()
        self.l1 = torch.nn.Linear(d_model, dff)
        self.l2 = torch.nn.Linear(dff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))

class EncoderBlock(nn.Module):
    """ Og transformer decoder block"""
    def __init__(self, d_model=512, n_heads=8, dff=2048):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model, n_heads, False)
        self.ffn = FFN(dff, d_model) 
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attention(x))
        x = self.ln2(x + self.ffn(x)) 
        return x

class DecoderBlock(nn.Module):
    """ Og transformer decoder block"""
    def __init__(self, d_model=512, n_heads=8, dff=2048):
        super().__init__()
        self.cross_attn = MultiHeadedAttention(d_model, n_heads, False)
        self.masked_attn = MultiHeadedAttention(d_model, n_heads, True)
        self.ffn = FFN(dff, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model) 
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output):
        x = self.ln1(x + self.masked_attn(x)) 
        x = self.ln2(x + self.cross_attn(x, enc_output)) 
        x = self.ln3(x + self.ffn(x))
        return x

class PosEncoding(nn.Module):
    def __init__(self, d_model, block_size):
        """block size: max length of a sequence"""
        super().__init__()
        self.pos_embedding = nn.Embedding(block_size, d_model)

    def forward(self, x_bld):
        l = x_bld.size(1)
        positions = torch.arange(l, device=x_bld.device)
        pos_emb = self.pos_embedding(positions)
        return x_bld + pos_emb 
          
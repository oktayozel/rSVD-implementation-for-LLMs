"""
Multi-Head Latent Attention (MLA) implementation.

Based on DeepSeek-V2 paper: https://arxiv.org/abs/2405.04434
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import time

from mla_gpt.model.compression.svd_compression import SVDCompression
from mla_gpt.model.compression.randomized_svd_compression import RandomizedSVDCompression

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention with optional SVD compression.
    
    MLA compresses KV cache by projecting to low-dimensional latent space.
    Optional SVD can be applied dynamically to KV latent representations.
    """

    def __init__(self, config):
        super().__init__()
        
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # MLA latent dimensions
        self.kv_latent_dim = getattr(config, 'kv_latent_dim', config.n_embd // 4)
        self.q_latent_dim = getattr(config, 'q_latent_dim', config.n_embd // 2)
        self.use_rope = getattr(config, 'use_rope', False)
        
        # SVD compression configuration
        self.kv_compression_type = getattr(config, 'kv_compression_type', 'none')
        self.kv_compression_rank = getattr(config, 'kv_compression_rank', None)
        
        # Initialize SVD compressors if needed
        self.kv_compressor = None
        if self.kv_compression_type != 'none' and self.kv_compression_rank is not None:
            self._init_kv_compressor(config)
        
        # Compression projections (down-projection compresses, up-projections decompress)
        self.kv_down_proj = nn.Linear(config.n_embd, self.kv_latent_dim, bias=config.bias)
        
        # Separate up-projections for K and V
        self.k_up_proj = nn.Linear(self.kv_latent_dim, config.n_embd, bias=config.bias)
        self.v_up_proj = nn.Linear(self.kv_latent_dim, config.n_embd, bias=config.bias)
        
        # Query compression
        self.q_down_proj = nn.Linear(config.n_embd, self.q_latent_dim, bias=config.bias)
        self.q_up_proj = nn.Linear(self.q_latent_dim, config.n_embd, bias=config.bias)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def _init_kv_compressor(self, config):
        """Initialize SVD compressor for KV latent compression."""
        print(f"Initializing MLA with {self.kv_compression_type} compression")
        print(f"  KV latent dim: {self.kv_latent_dim}")
        print(f"  Compression rank: {self.kv_compression_rank}")
        
        if self.kv_compression_type == 'svd':
            self.kv_compressor = SVDCompression(rank=self.kv_compression_rank)
        elif self.kv_compression_type == 'randomized_svd':
            n_oversamples = getattr(config, 'svd_n_oversamples', 10)
            n_power_iter = getattr(config, 'svd_n_power_iter', 2)
            self.kv_compressor = RandomizedSVDCompression(
                rank=self.kv_compression_rank,
                n_oversamples=n_oversamples,
                n_power_iter=n_power_iter
            )
        else:
            raise ValueError(f"Unknown compression type: {self.kv_compression_type}")

    def forward(self, x, kv_cache=None, use_cache=False):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        
        # === Query path ===
        # Compress query to latent space and decompress
        q_latent = self.q_down_proj(x)  # (B, T, q_latent_dim)
        q = self.q_up_proj(q_latent)    # (B, T, n_embd)
        
        # === Key-Value path with optional dynamic SVD compression ===
        # Step 1: Compress to KV latent space
        kv_latent = self.kv_down_proj(x)  # (B, T, kv_latent_dim)
        
        # Step 2: Apply dynamic SVD compression if enabled
        if self.kv_compressor is not None:
            # Reshape for SVD: (B, T, kv_latent_dim) -> (B, kv_latent_dim, T) for column-wise compression
            kv_latent_t = kv_latent.transpose(1, 2)  # (B, kv_latent_dim, T)
            
            # Apply SVD compression (this happens EVERY forward pass!)
            kv_latent_compressed = self.kv_compressor.compress(kv_latent_t)  # (B, kv_latent_dim, T)
            
            # Reshape back
            kv_latent = kv_latent_compressed.transpose(1, 2)  # (B, T, kv_latent_dim)
        
        # Step 3: Decompress to get keys and values
        k = self.k_up_proj(kv_latent)  # (B, T, n_embd)
        v = self.v_up_proj(kv_latent)  # (B, T, n_embd)
        
        # === Reshape for multi-head attention ===
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # === Causal self-attention ===
        # Efficient attention using Flash Attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1))))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, hs)
        
        # Reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y
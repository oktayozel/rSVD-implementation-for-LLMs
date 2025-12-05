"""
Multi-Head Latent Attention (MLA) implementation.

Based on DeepSeek-V2 paper: https://arxiv.org/abs/2405.04434

MLA compresses the KV cache by projecting keys and values into a low-dimensional
latent space, significantly reducing memory requirements during inference while
maintaining model quality.

Key idea:
- Instead of caching full K and V matrices, cache a compressed latent representation
- K and V are reconstructed from the latent representation when needed
- This reduces KV cache memory from O(n_head * head_size) to O(latent_dim)
"""

import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from mla_gpt.model.compression.svd_compression import SVDCompression
from mla_gpt.model.compression.randomized_svd_compression import RandomizedSVDCompression
from mla_gpt.model.compression.svd_compression import SVDCompression
from mla_gpt.model.compression.randomized_svd_compression import RandomizedSVDCompression


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) as introduced in DeepSeek-V2.
    
    MLA uses low-rank joint compression for keys and values:
    1. Project input to a low-dimensional latent space (compression)
    2. Project latent to keys and values (decompression)
    3. This allows caching only the compressed latent, reducing KV cache size
    
    Args:
        config: Model configuration with the following attributes:
            - n_embd: Embedding dimension
            - n_head: Number of attention heads  
            - dropout: Dropout probability
            - bias: Whether to use bias in linear layers
            - block_size: Maximum sequence length
            - kv_latent_dim: Dimension of the KV latent space (optional, defaults to n_embd // 4)
            - q_latent_dim: Dimension of the Q latent space (optional, defaults to n_embd // 2)
            - use_rope: Whether to use rotary position embeddings for a portion of head dim (optional)
            - rope_dim: Dimension for RoPE if used (optional, defaults to head_size // 2)
            - kv_compression_type: Type of compression for KV projection ('none', 'svd', 'randomized_svd')
            - kv_compression_rank: Rank for KV projection compression
            - svd_n_oversamples: Oversamples for randomized SVD (default: 10)
            - svd_n_power_iter: Power iterations for randomized SVD (default: 2)
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Latent dimensions for compression
        # KV latent dim controls KV cache compression ratio
        self.kv_latent_dim = getattr(config, 'kv_latent_dim', config.n_embd // 4)
        # Q latent dim for query compression (optional, can match n_embd for no compression)
        self.q_latent_dim = getattr(config, 'q_latent_dim', config.n_embd // 2)
        
        # RoPE settings (optional decoupled RoPE as in DeepSeek-V2)
        self.use_rope = getattr(config, 'use_rope', False)
        self.rope_dim = getattr(config, 'rope_dim', self.head_size // 2) if self.use_rope else 0
        
        # ============ Query Path ============
        # Down-projection: x -> q_latent
        self.q_down_proj = nn.Linear(config.n_embd, self.q_latent_dim, bias=config.bias)
        # Up-projection: q_latent -> queries (n_head * head_size)
        self.q_up_proj = nn.Linear(self.q_latent_dim, config.n_embd, bias=config.bias)
        
        # Optional: separate projection for RoPE portion of queries
        if self.use_rope:
            self.q_rope_proj = nn.Linear(config.n_embd, self.n_head * self.rope_dim, bias=config.bias)
        
        # ============ KV Path (Joint Compression) ============
        # Compression settings
        self.kv_compression_type = getattr(config, 'kv_compression_type', 'none')
        self.kv_compression_rank = getattr(config, 'kv_compression_rank', None)
        self.compression_time = 0.0  # Track compression time
        self.reconstruction_error = 0.0  # Track reconstruction error
        
        # Down-projection: x -> kv_latent (this is what gets cached!)
        self.kv_down_proj = nn.Linear(config.n_embd, self.kv_latent_dim, bias=config.bias)
        
        # Apply SVD compression to kv_down_proj weights if requested
        if self.kv_compression_type != 'none' and self.kv_compression_rank is not None:
            self._apply_kv_compression(config)
        
        # Up-projections: kv_latent -> keys, values
        self.k_up_proj = nn.Linear(self.kv_latent_dim, config.n_embd, bias=config.bias)
        self.v_up_proj = nn.Linear(self.kv_latent_dim, config.n_embd, bias=config.bias)
        
        # Optional: separate projection for RoPE portion of keys
        if self.use_rope:
            self.k_rope_proj = nn.Linear(config.n_embd, self.n_head * self.rope_dim, bias=config.bias)
        
        # ============ Output Projection ============
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias", 
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )
        
        # Precompute RoPE frequencies if using RoPE
        if self.use_rope:
            self._init_rope(config.block_size)
    
    def _init_rope(self, max_seq_len, base=10000.0):
        """Initialize rotary position embedding frequencies."""
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for all positions
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().view(1, 1, max_seq_len, self.rope_dim))
        self.register_buffer("sin_cached", emb.sin().view(1, 1, max_seq_len, self.rope_dim))
    
    def _apply_kv_compression(self, config):
        """Apply SVD compression to KV down-projection weights."""
        start_time = time.time()
        
        # Get original weight matrix
        original_weight = self.kv_down_proj.weight.data.clone()
        
        # Create compression object
        if self.kv_compression_type == 'svd':
            compressor = SVDCompression(
                rank=self.kv_compression_rank,
                compression_type='standard'
            )
        elif self.kv_compression_type == 'randomized_svd':
            compressor = RandomizedSVDCompression(
                rank=self.kv_compression_rank,
                n_oversamples=getattr(config, 'svd_n_oversamples', 10),
                n_power_iter=getattr(config, 'svd_n_power_iter', 2)
            )
        else:
            return  # No compression
        
        # Apply compression
        compressed_weight = compressor.compress(original_weight)
        
        # Replace weights
        self.kv_down_proj.weight.data = compressed_weight
        
        # Track metrics
        self.compression_time = time.time() - start_time
        self.reconstruction_error = torch.norm(original_weight - compressed_weight).item() / torch.norm(original_weight).item()
        
        print(f"  KV compression: {self.kv_compression_type}, rank={self.kv_compression_rank}")
        print(f"  Compression time: {self.compression_time:.4f}s")
        print(f"  Reconstruction error: {self.reconstruction_error:.6f}")
    
    def _apply_rope(self, x, seq_len):
        """Apply rotary position embeddings to x."""
        # x: (B, nh, T, rope_dim)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        # Rotate half
        x1 = x[..., :self.rope_dim // 2]
        x2 = x[..., self.rope_dim // 2:]
        x_rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos + x_rotated * sin
    
    def forward(self, x, kv_cache=None, use_cache=False):
        """
        Forward pass for Multi-Head Latent Attention.
        
        Args:
            x: Input tensor of shape (B, T, C)
            kv_cache: Optional cached KV latent from previous forward pass
            use_cache: Whether to return the KV latent for caching
            
        Returns:
            y: Output tensor of shape (B, T, C)
            new_kv_cache: KV latent for caching (only if use_cache=True)
        """
        B, T, C = x.size()
        
        # ============ Query Computation ============
        # Compress then decompress queries
        q_latent = self.q_down_proj(x)  # (B, T, q_latent_dim)
        q = self.q_up_proj(q_latent)     # (B, T, n_embd)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        
        # ============ KV Computation with Latent Compression ============
        # Compress to latent space (this is what we cache!)
        kv_latent = self.kv_down_proj(x)  # (B, T, kv_latent_dim)
        
        # Handle caching
        if kv_cache is not None:
            # Concatenate with cached latent
            kv_latent = torch.cat([kv_cache, kv_latent], dim=1)
        
        # Decompress to get keys and values
        k = self.k_up_proj(kv_latent)  # (B, T_kv, n_embd)
        v = self.v_up_proj(kv_latent)  # (B, T_kv, n_embd)
        
        T_kv = kv_latent.size(1)  # May be longer than T if using cache
        
        k = k.view(B, T_kv, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T_kv, hs)
        v = v.view(B, T_kv, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T_kv, hs)
        
        # ============ Optional RoPE ============
        if self.use_rope:
            # Get RoPE components for Q and K
            q_rope = self.q_rope_proj(x)  # (B, T, n_head * rope_dim)
            q_rope = q_rope.view(B, T, self.n_head, self.rope_dim).transpose(1, 2)
            
            # For K, we need the original x before caching was applied
            # In practice, we'd handle this more carefully for incremental decoding
            k_rope = self.k_rope_proj(x)  # (B, T, n_head * rope_dim)
            k_rope = k_rope.view(B, T, self.n_head, self.rope_dim).transpose(1, 2)
            
            # Apply RoPE
            q_rope = self._apply_rope(q_rope, T)
            k_rope = self._apply_rope(k_rope, T)
            
            # Concatenate RoPE and non-RoPE portions
            # For simplicity, we replace the first rope_dim dimensions
            q = torch.cat([q_rope, q[..., self.rope_dim:]], dim=-1)
            k = torch.cat([k_rope, k[..., self.rope_dim:]], dim=-1)
        
        # ============ Attention Computation ============
        if self.flash:
            # Use Flash Attention
            # Note: For cached KV, we need is_causal=False and provide proper mask
            if kv_cache is not None:
                # During incremental decoding, no causal mask needed for new tokens
                y = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None, 
                    dropout_p=self.dropout if self.training else 0, 
                    is_causal=False
                )
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None, 
                    dropout_p=self.dropout if self.training else 0, 
                    is_causal=True
                )
        else:
            # Manual attention implementation
            scale = 1.0 / math.sqrt(self.head_size)
            att = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T_kv)
            
            if kv_cache is None:
                # Apply causal mask during training/prefill
                att = att.masked_fill(self.bias[:, :, :T, :T_kv] == 0, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, hs)
        
        # ============ Output ============
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        if use_cache:
            return y, kv_latent
        return y
    
    def get_kv_cache_size(self):
        """
        Returns the KV cache size per token.
        
        Standard MHA caches: 2 * n_head * head_size = 2 * n_embd
        MLA caches: kv_latent_dim
        
        Compression ratio: 2 * n_embd / kv_latent_dim
        """
        standard_cache_size = 2 * self.n_embd
        mla_cache_size = self.kv_latent_dim
        compression_ratio = standard_cache_size / mla_cache_size
        
        return {
            'standard_mha_cache_per_token': standard_cache_size,
            'mla_cache_per_token': mla_cache_size,
            'compression_ratio': compression_ratio
        }
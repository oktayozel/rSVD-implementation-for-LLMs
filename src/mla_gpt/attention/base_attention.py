"""
Base attention class with common functionality
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class BaseAttention(nn.Module):
    """
    Base class for attention mechanisms with common functionality
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_size = config.n_embd // config.n_head
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask for non-flash attention
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
    
    def _apply_flash_attention(self, q, k, v):
        """Apply flash attention if available"""
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0, 
            is_causal=True
        )
    
    def _apply_manual_attention(self, q, k, v):
        """Manual attention computation with causal masking"""
        B, nh, T, hs = q.shape
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / (hs ** 0.5))
        
        # Apply causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        if hasattr(self, 'attn_dropout'):
            att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        return y
    
    def forward(self, x):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward method")
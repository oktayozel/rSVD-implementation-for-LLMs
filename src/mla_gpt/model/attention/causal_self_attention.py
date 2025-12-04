import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # SVD parameters - separate for Q, K, V
        # Backwards compatibility: use_svd applies to V only
        self.use_svd_q = getattr(config, 'use_svd_q', False)
        self.use_svd_k = getattr(config, 'use_svd_k', False)
        self.use_svd_v = getattr(config, 'use_svd_v', False) or getattr(config, 'use_svd', False)
        
        # Ranks for each matrix
        default_rank = getattr(config, 'svd_rank', None)
        self.svd_rank_q = getattr(config, 'svd_rank_q', default_rank)
        self.svd_rank_k = getattr(config, 'svd_rank_k', default_rank)
        self.svd_rank_v = getattr(config, 'svd_rank_v', default_rank)
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def apply_svd(self, matrix, use_svd, svd_rank, matrix_name=""):
        """
        Apply SVD to any Q, K, or V matrix
        
        Args:
            matrix: Tensor of shape (B, nh, T, hs)
            use_svd: Whether to apply SVD to this matrix
            svd_rank: Rank to use for this matrix
            matrix_name: Name for logging (Q/K/V)
            
        Returns:
            Reconstructed matrix after SVD decomposition (or original if SVD disabled)
        """
        if not use_svd:
            return matrix
        
        return self._standard_svd_reconstruction(matrix, svd_rank)

    def _standard_svd_reconstruction(self, matrix, rank):
        """Standard SVD reconstruction using torch.linalg.svd"""
        B, nh, T, hs = matrix.shape
        
        # Reshape matrix to (B*nh, T, hs) for SVD computation
        matrix_reshaped = matrix.reshape(B * nh, T, hs)
        
        # Initialize output tensor
        matrix_reconstructed = torch.zeros_like(matrix_reshaped)
        
        # Apply SVD to each (T, hs) matrix in the batch
        for i in range(B * nh):
            # Perform SVD: Matrix = U @ S @ Vh
            U, S, Vh = torch.linalg.svd(matrix_reshaped[i], full_matrices=False)
            
            # Determine rank for reconstruction
            if rank is not None:
                r = min(rank, S.shape[0])
            else:
                r = S.shape[0]  # Full rank reconstruction
            
            # Reconstruct with reduced rank
            # Matrix_approx = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]
            S_diag = torch.diag(S[:r])
            matrix_reconstructed[i] = U[:, :r] @ S_diag @ Vh[:r, :]
        
        # Reshape back to original dimensions
        return matrix_reconstructed.reshape(B, nh, T, hs)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Apply SVD to Q, K, V matrices independently
        q = self.apply_svd(q, self.use_svd_q, self.svd_rank_q, "Q")
        k = self.apply_svd(k, self.use_svd_k, self.svd_rank_k, "K")
        v = self.apply_svd(v, self.use_svd_v, self.svd_rank_v, "V")

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
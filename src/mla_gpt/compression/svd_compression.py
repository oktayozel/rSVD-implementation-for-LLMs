"""
SVD-based matrix compression
"""

import torch
import torch.nn as nn
from .base_compression import BaseCompression


class SVDCompression(BaseCompression):
    """
    Singular Value Decomposition (SVD) based matrix compression
    
    Supports both standard SVD and future randomized SVD implementations
    """
    
    def __init__(self, rank=None, compression_type='standard', **kwargs):
        super().__init__(rank=rank, **kwargs)
        self.compression_type = compression_type
        
        if compression_type not in ['standard', 'randomized']:
            raise ValueError(f"Unknown compression_type: {compression_type}")
    
    def compress(self, matrix):
        """
        Apply SVD compression to the input matrix
        
        Args:
            matrix: Input tensor of any shape. SVD will be applied to the last two dimensions
            
        Returns:
            Compressed tensor of same shape
        """
        if self.compression_type == 'standard':
            return self._standard_svd_compression(matrix)
        elif self.compression_type == 'randomized':
            return self._randomized_svd_compression(matrix)
    
    def _standard_svd_compression(self, matrix):
        """
        Standard SVD compression using torch.linalg.svd
        
        Args:
            matrix: Tensor of shape (..., M, N) where SVD is applied to (M, N) matrices
            
        Returns:
            Compressed tensor of same shape
        """
        original_shape = matrix.shape
        
        # Handle different input shapes
        if len(original_shape) == 2:
            # Single matrix (M, N)
            return self._compress_single_matrix(matrix)
        elif len(original_shape) == 3:
            # Batch of matrices (B, M, N)
            B, M, N = original_shape
            compressed = torch.zeros_like(matrix)
            for i in range(B):
                compressed[i] = self._compress_single_matrix(matrix[i])
            return compressed
        elif len(original_shape) == 4:
            # Common case for attention: (B, nh, T, hs)
            B, nh, T, hs = original_shape
            matrix_reshaped = matrix.reshape(B * nh, T, hs)
            compressed_reshaped = torch.zeros_like(matrix_reshaped)
            
            for i in range(B * nh):
                compressed_reshaped[i] = self._compress_single_matrix(matrix_reshaped[i])
            
            return compressed_reshaped.reshape(original_shape)
        else:
            # General case: flatten all but last two dimensions
            *batch_dims, M, N = original_shape
            batch_size = 1
            for dim in batch_dims:
                batch_size *= dim
            
            matrix_reshaped = matrix.reshape(batch_size, M, N)
            compressed_reshaped = torch.zeros_like(matrix_reshaped)
            
            for i in range(batch_size):
                compressed_reshaped[i] = self._compress_single_matrix(matrix_reshaped[i])
            
            return compressed_reshaped.reshape(original_shape)
    
    def _compress_single_matrix(self, matrix):
        """
        Compress a single 2D matrix using SVD
        
        Args:
            matrix: 2D tensor of shape (M, N)
            
        Returns:
            Compressed 2D tensor of shape (M, N)
        """
        # Perform SVD: Matrix = U @ S @ Vh
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        
        # Determine rank for reconstruction
        if self.rank is not None:
            rank = min(self.rank, S.shape[0])
        else:
            rank = S.shape[0]  # Full rank reconstruction
        
        # Reconstruct with reduced rank
        # Matrix_approx = U[:, :rank] @ diag(S[:rank]) @ Vh[:rank, :]
        S_diag = torch.diag(S[:rank])
        compressed_matrix = U[:, :rank] @ S_diag @ Vh[:rank, :]
        
        return compressed_matrix
    
    def _randomized_svd_compression(self, matrix):
        """
        Randomized SVD compression (placeholder for future implementation)
        
        Args:
            matrix: Input tensor
            
        Returns:
            Compressed tensor (currently falls back to standard SVD)
        """
        # TODO: Implement randomized SVD
        # For now, fall back to standard SVD
        print("WARNING: Randomized SVD not yet implemented. Using standard SVD.")
        return self._standard_svd_compression(matrix)
    
    def get_compression_ratio(self, matrix_shape, rank=None):
        """
        Calculate the compression ratio for given matrix shape and rank
        
        Args:
            matrix_shape: Tuple representing the matrix shape (..., M, N)
            rank: Compression rank (uses self.rank if None)
            
        Returns:
            Compression ratio (original_params / compressed_params)
        """
        if rank is None:
            rank = self.rank
        
        if rank is None:
            return 1.0  # No compression
        
        *batch_dims, M, N = matrix_shape
        
        # Original parameters
        original_params = M * N
        
        # Compressed parameters: U(:, :rank) + S(:rank) + Vh(:rank, :)
        compressed_params = M * rank + rank + rank * N
        
        return original_params / compressed_params
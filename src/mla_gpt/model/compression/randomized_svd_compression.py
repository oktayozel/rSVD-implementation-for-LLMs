"""
Randomized SVD compression implementation for attention matrices.

Based on "Finding Structure with Randomness: Probabilistic Algorithms for 
Constructing Approximate Matrix Decompositions" (Halko, Martinsson, Tropp, 2011)

Key advantages over standard SVD:
- Faster computation for large matrices
- Lower memory footprint
- Tunable accuracy vs speed trade-off
"""

import torch
import torch.nn as nn
from .base_compression import BaseCompression


class RandomizedSVDCompression(BaseCompression):
    """
    Randomized SVD compression for efficient low-rank approximation.
    
    Algorithm:
    1. Draw random test matrix Omega
    2. Compute Y = A @ Omega (range finder)
    3. Orthonormalize Y to get Q
    4. Compute B = Q.T @ A
    5. Compute SVD of smaller matrix B
    6. Reconstruct: A_approx = Q @ U @ S @ Vh
    
    Args:
        rank: Target rank for approximation
        n_oversamples: Additional samples for accuracy (default: 10)
        n_power_iter: Number of power iterations for accuracy (default: 2)
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, rank=None, n_oversamples=10, n_power_iter=2, 
                 random_state=None, **kwargs):
        super().__init__(rank=rank, **kwargs)
        self.n_oversamples = n_oversamples
        self.n_power_iter = n_power_iter
        self.random_state = random_state
        
        if self.rank is None:
            raise ValueError("rank must be specified for randomized SVD")
    
    def compress(self, matrix):
        """
        Apply randomized SVD compression to input matrix.
        
        Args:
            matrix: Input tensor of shape (..., M, N)
            
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
        Compress a single 2D matrix using randomized SVD.
        
        Args:
            matrix: 2D tensor of shape (M, N)
            
        Returns:
            Compressed 2D tensor of shape (M, N)
        """
        M, N = matrix.shape
        
        # Determine effective rank (rank + oversampling)
        k = min(self.rank, min(M, N))
        n_random = min(k + self.n_oversamples, min(M, N))
        
        # Step 1: Randomized range finding
        Q = self._randomized_range_finder(
            matrix, n_random, self.n_power_iter
        )
        
        # Step 2: Compute B = Q.T @ A (project to lower dimension)
        B = Q.T @ matrix  # (n_random, N)
        
        # Step 3: Compute SVD of smaller matrix B
        U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
        
        # Truncate to desired rank
        U_tilde = U_tilde[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]
        
        # Step 4: Recover U = Q @ U_tilde
        U = Q @ U_tilde
        
        # Step 5: Reconstruct matrix
        compressed_matrix = U @ torch.diag(S) @ Vh
        
        return compressed_matrix
    
    def _randomized_range_finder(self, A, size, n_iter):
        """
        Compute an approximate orthonormal basis for the range of A.
        
        This is the core randomized algorithm that finds a good subspace
        for approximating the matrix.
        
        Args:
            A: Input matrix (M, N)
            size: Size of the random subspace
            n_iter: Number of power iterations
            
        Returns:
            Q: Orthonormal basis matrix (M, size)
        """
        M, N = A.shape
        
        # Generate random test matrix
        if self.random_state is not None:
            generator = torch.Generator(device=A.device)
            generator.manual_seed(self.random_state)
            Omega = torch.randn(N, size, device=A.device, dtype=A.dtype, 
                               generator=generator)
        else:
            Omega = torch.randn(N, size, device=A.device, dtype=A.dtype)
        
        # Initial projection
        Y = A @ Omega  # (M, size)
        
        # Power iterations (optional, improves accuracy)
        for _ in range(n_iter):
            # Orthonormalize Y
            Y, _ = torch.linalg.qr(Y)
            # Apply A.T and A
            Z = A.T @ Y  # (N, size)
            Z, _ = torch.linalg.qr(Z)
            Y = A @ Z  # (M, size)
        
        # Final orthonormalization
        Q, _ = torch.linalg.qr(Y)
        
        return Q
    
    def get_compression_ratio(self, matrix_shape, rank=None):
        """
        Calculate compression ratio for randomized SVD.
        
        Args:
            matrix_shape: Tuple (M, N)
            rank: Compression rank (uses self.rank if None)
            
        Returns:
            Compression ratio
        """
        if rank is None:
            rank = self.rank
        
        if rank is None:
            return 1.0
        
        *batch_dims, M, N = matrix_shape
        
        # Original parameters
        original_params = M * N
        
        # Compressed representation: U + S + Vh
        # U: (M, rank), S: (rank,), Vh: (rank, N)
        compressed_params = M * rank + rank + rank * N
        
        return original_params / compressed_params
    
    def get_computational_complexity(self, matrix_shape):
        """
        Estimate computational complexity of randomized SVD.
        
        Standard SVD: O(min(M*N^2, M^2*N))
        Randomized SVD: O(M*N*k + (M+N)*k^2) where k = rank + oversamples
        
        Args:
            matrix_shape: Tuple (M, N)
            
        Returns:
            Dictionary with complexity estimates
        """
        *batch_dims, M, N = matrix_shape
        k = min(self.rank + self.n_oversamples, min(M, N))
        
        # Randomized SVD complexity
        range_finding = M * N * k  # A @ Omega
        power_iter = 2 * self.n_power_iter * M * N * k  # Power iterations
        qr_factorization = M * k * k  # QR decomposition
        small_svd = k * k * min(k, N)  # SVD of B
        
        randomized_complexity = (range_finding + power_iter + 
                                qr_factorization + small_svd)
        
        # Standard SVD complexity
        standard_complexity = min(M * N * N, M * M * N)
        
        return {
            'randomized_svd_flops': randomized_complexity,
            'standard_svd_flops': standard_complexity,
            'speedup_ratio': standard_complexity / randomized_complexity,
            'rank': self.rank,
            'effective_rank': k,
            'n_power_iter': self.n_power_iter
        }


class AdaptiveRandomizedSVDCompression(RandomizedSVDCompression):
    """
    Adaptive randomized SVD that adjusts rank based on singular value decay.
    
    This variant automatically determines the effective rank by analyzing
    the singular value spectrum and keeping only significant components.
    
    Args:
        rank: Maximum rank
        energy_threshold: Keep singular values that capture this fraction 
                         of total energy (default: 0.95)
        min_rank: Minimum rank to keep (default: 1)
    """
    
    def __init__(self, rank=None, energy_threshold=0.95, min_rank=1, 
                 n_oversamples=10, n_power_iter=2, random_state=None, **kwargs):
        super().__init__(rank=rank, n_oversamples=n_oversamples,
                        n_power_iter=n_power_iter, random_state=random_state, 
                        **kwargs)
        self.energy_threshold = energy_threshold
        self.min_rank = min_rank
    
    def _compress_single_matrix(self, matrix):
        """
        Compress with adaptive rank selection based on energy.
        """
        M, N = matrix.shape
        
        # Determine effective rank
        k = min(self.rank, min(M, N))
        n_random = min(k + self.n_oversamples, min(M, N))
        
        # Randomized range finding
        Q = self._randomized_range_finder(matrix, n_random, self.n_power_iter)
        
        # Compute B = Q.T @ A
        B = Q.T @ matrix
        
        # Compute SVD of smaller matrix
        U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
        
        # Adaptive rank selection based on energy
        total_energy = torch.sum(S ** 2)
        cumulative_energy = torch.cumsum(S ** 2, dim=0)
        energy_ratio = cumulative_energy / total_energy
        
        # Find rank that captures desired energy
        adaptive_rank = torch.searchsorted(energy_ratio, self.energy_threshold).item() + 1
        adaptive_rank = max(self.min_rank, min(adaptive_rank, k))
        
        # Truncate to adaptive rank
        U_tilde = U_tilde[:, :adaptive_rank]
        S = S[:adaptive_rank]
        Vh = Vh[:adaptive_rank, :]
        
        # Recover U and reconstruct
        U = Q @ U_tilde
        compressed_matrix = U @ torch.diag(S) @ Vh
        
        return compressed_matrix
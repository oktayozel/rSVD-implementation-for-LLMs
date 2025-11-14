"""
Base compression class for matrix compression algorithms
"""

import torch
import torch.nn as nn


class BaseCompression(nn.Module):
    """
    Base class for matrix compression algorithms
    """
    
    def __init__(self, rank=None, **kwargs):
        super().__init__()
        self.rank = rank
        
    def compress(self, matrix):
        """
        Compress the input matrix
        
        Args:
            matrix: Input tensor to compress
            
        Returns:
            Compressed tensor of same shape
        """
        raise NotImplementedError("Subclasses must implement compress method")
    
    def forward(self, matrix):
        """Forward pass - alias for compress"""
        return self.compress(matrix)
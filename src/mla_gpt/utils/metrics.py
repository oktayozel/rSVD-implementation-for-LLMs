"""
Focused Metrics Collection for MLA Experiments.

This module provides utilities for measuring:
1. Inference speed (tokens/sec)
2. Perplexity (model quality)
3. Compression error (reconstruction error of SVD)
4. Compression time (time to apply SVD)
"""

import time
import torch
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


# =============================================================================
# 1. Inference Speed
# =============================================================================

def measure_inference_speed(model, data_loader, num_batches=10, device='cuda'):
    """
    Measure inference speed in tokens per second.
    
    Args:
        model: The GPT model to evaluate
        data_loader: DataLoader providing (x, y) batches
        num_batches: Number of batches to measure
        device: Device to run on
        
    Returns:
        dict with inference metrics:
            - tokens_per_second: Inference speed
            - batches_per_second: Batch processing rate
            - total_time: Total elapsed time
            - total_tokens: Total tokens processed
    """
    model.eval()
    
    # Warmup (2 batches)
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if i >= 2:
                break
            x = x.to(device)
            logits, loss = model(x)
    
    # Actual measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    total_tokens = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            x = x.to(device)
            logits, loss = model(x)
            total_tokens += x.numel()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    tokens_per_second = total_tokens / elapsed_time
    batches_per_second = num_batches / elapsed_time
    
    return {
        'tokens_per_second': tokens_per_second,
        'batches_per_second': batches_per_second,
        'total_time': elapsed_time,
        'total_tokens': total_tokens,
    }


# =============================================================================
# 2. Perplexity
# =============================================================================

def compute_perplexity(model, data_loader, max_batches=None, device='cuda'):
    """
    Compute perplexity on a dataset.
    
    Perplexity = exp(average cross-entropy loss)
    Lower perplexity = better model
    
    Args:
        model: The GPT model to evaluate
        data_loader: DataLoader providing (x, y) batches
        max_batches: Maximum number of batches to evaluate (None for all)
        device: Device to run on
        
    Returns:
        tuple of (perplexity, avg_loss):
            - perplexity: The computed perplexity
            - avg_loss: Average loss across batches
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break
            
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            
            # Accumulate loss weighted by number of tokens
            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity, avg_loss


# =============================================================================
# 3. Compression Error (Reconstruction Error)
# =============================================================================

def compute_reconstruction_error(original, reconstructed, metric='relative'):
    """
    Compute reconstruction error between original and reconstructed matrices.
    
    Args:
        original: Original matrix tensor
        reconstructed: Reconstructed matrix tensor (must have same shape)
        metric: Error metric type:
            - 'relative': Relative Frobenius norm (default, scale-invariant)
            - 'frobenius': Absolute Frobenius norm
            - 'mse': Mean squared error
        
    Returns:
        float: Reconstruction error
    """
    assert original.shape == reconstructed.shape, "Matrices must have same shape"
    
    if metric == 'relative':
        # Relative Frobenius norm: ||A - B||_F / ||A||_F
        diff_norm = torch.norm(original - reconstructed, p='fro')
        orig_norm = torch.norm(original, p='fro')
        error = (diff_norm / orig_norm).item() if orig_norm > 0 else 0.0
        
    elif metric == 'frobenius':
        # Absolute Frobenius norm: ||A - B||_F
        error = torch.norm(original - reconstructed, p='fro').item()
        
    elif metric == 'mse':
        # Mean squared error
        error = F.mse_loss(reconstructed, original).item()
        
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'relative', 'frobenius', or 'mse'")
    
    return error


def extract_compression_metrics_from_model(model):
    """
    Extract compression metrics from an MLA model with KV compression.
    
    This function looks for MLA attention modules that have been compressed
    and extracts their compression error and compression time.
    
    Args:
        model: GPT model with MLA attention
        
    Returns:
        dict with aggregated compression metrics:
            - avg_reconstruction_error: Average reconstruction error across layers
            - avg_compression_time: Average compression time across layers
            - total_compression_time: Total compression time
            - num_compressed_layers: Number of layers with compression
            - compression_type: Type of compression used ('none', 'svd', 'randomized_svd')
            - compression_rank: Rank used for compression
    """
    reconstruction_errors = []
    compression_times = []
    compression_type = 'none'
    compression_rank = None
    
    # Iterate through transformer blocks
    for block in model.transformer.h:
        if hasattr(block.attn, 'kv_compression_type'):
            # This is an MLA attention module
            compression_type = block.attn.kv_compression_type
            
            if hasattr(block.attn, 'kv_compression_rank'):
                compression_rank = block.attn.kv_compression_rank
            
            if hasattr(block.attn, 'reconstruction_error'):
                reconstruction_errors.append(block.attn.reconstruction_error)
            
            if hasattr(block.attn, 'compression_time'):
                compression_times.append(block.attn.compression_time)
    
    num_compressed_layers = len(reconstruction_errors)
    
    metrics = {
        'compression_type': compression_type,
        'compression_rank': compression_rank,
        'num_compressed_layers': num_compressed_layers,
    }
    
    if reconstruction_errors:
        metrics['avg_reconstruction_error'] = np.mean(reconstruction_errors)
        metrics['min_reconstruction_error'] = np.min(reconstruction_errors)
        metrics['max_reconstruction_error'] = np.max(reconstruction_errors)
        metrics['std_reconstruction_error'] = np.std(reconstruction_errors)
    else:
        metrics['avg_reconstruction_error'] = 0.0
        metrics['min_reconstruction_error'] = 0.0
        metrics['max_reconstruction_error'] = 0.0
        metrics['std_reconstruction_error'] = 0.0
    
    if compression_times:
        metrics['avg_compression_time'] = np.mean(compression_times)
        metrics['total_compression_time'] = np.sum(compression_times)
        metrics['min_compression_time'] = np.min(compression_times)
        metrics['max_compression_time'] = np.max(compression_times)
    else:
        metrics['avg_compression_time'] = 0.0
        metrics['total_compression_time'] = 0.0
        metrics['min_compression_time'] = 0.0
        metrics['max_compression_time'] = 0.0
    
    return metrics


# =============================================================================
# 4. Compression Time (already captured during model initialization)
# =============================================================================

# Compression time is measured during model initialization when SVD is applied
# to the KV down-projection weights. It's stored in the attention module and
# can be retrieved using extract_compression_metrics_from_model()


# =============================================================================
# Additional Useful Metrics
# =============================================================================

def compute_model_size(model):
    """
    Compute model size in parameters and memory.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict with size metrics:
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters
            - size_mb: Model size in megabytes (float32)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Assuming float32 (4 bytes per parameter)
    size_mb = total_params * 4 / (1024 ** 2)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': size_mb,
    }


def compute_flops_per_token(config):
    """
    Estimate FLOPs per token for a Transformer model.
    
    This is a rough estimate based on the model architecture.
    
    Args:
        config: GPTConfig object
        
    Returns:
        float: Estimated FLOPs per token
    """
    # Simplified FLOP calculation for Transformer
    # Each layer has: attention (QKV projections + scores + output) + FFN
    
    n_embd = config.n_embd
    n_layer = config.n_layer
    block_size = config.block_size
    
    # Attention FLOPs per layer per token
    # QKV projections: 3 * 2 * n_embd^2
    # Attention scores: 2 * n_embd * block_size
    # Output projection: 2 * n_embd^2
    attn_flops = 3 * 2 * n_embd**2 + 2 * n_embd * block_size + 2 * n_embd**2
    
    # FFN FLOPs per layer per token
    # Up projection: 2 * n_embd * 4 * n_embd
    # Down projection: 2 * 4 * n_embd * n_embd
    ffn_flops = 2 * n_embd * 4 * n_embd + 2 * 4 * n_embd * n_embd
    
    # Total per token
    total_flops_per_token = n_layer * (attn_flops + ffn_flops)
    
    return total_flops_per_token


# =============================================================================
# Comprehensive Evaluation Function
# =============================================================================

def evaluate_model_comprehensive(model, data_loader, device='cuda', 
                                 num_inference_batches=10, num_ppl_batches=50):
    """
    Comprehensive evaluation collecting all key metrics.
    
    Args:
        model: GPT model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to run on
        num_inference_batches: Number of batches for inference speed measurement
        num_ppl_batches: Number of batches for perplexity computation
        
    Returns:
        dict with all metrics:
            - inference_speed_metrics: tokens/sec, etc.
            - perplexity_metrics: perplexity and loss
            - compression_metrics: reconstruction error, compression time
            - model_size_metrics: parameters and memory
    """
    results = {}
    
    # 1. Inference Speed
    print("Measuring inference speed...")
    results['inference_speed'] = measure_inference_speed(
        model, data_loader, num_batches=num_inference_batches, device=device
    )
    
    # 2. Perplexity
    print("Computing perplexity...")
    perplexity, avg_loss = compute_perplexity(
        model, data_loader, max_batches=num_ppl_batches, device=device
    )
    results['perplexity'] = perplexity
    results['avg_loss'] = avg_loss
    
    # 3. Compression Metrics
    print("Extracting compression metrics...")
    results['compression'] = extract_compression_metrics_from_model(model)
    
    # 4. Model Size
    results['model_size'] = compute_model_size(model)
    
    # 5. FLOPs estimate
    results['estimated_flops_per_token'] = compute_flops_per_token(model.config)
    
    return results


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_models(model1_results, model2_results, model1_name='Model 1', model2_name='Model 2'):
    """
    Compare two model evaluation results.
    
    Args:
        model1_results: Results dict from evaluate_model_comprehensive for model 1
        model2_results: Results dict from evaluate_model_comprehensive for model 2
        model1_name: Name for model 1
        model2_name: Name for model 2
        
    Returns:
        dict with comparison metrics showing improvements/degradations
    """
    comparison = {}
    
    # Inference speed comparison (higher is better)
    speed1 = model1_results['inference_speed']['tokens_per_second']
    speed2 = model2_results['inference_speed']['tokens_per_second']
    comparison['inference_speedup'] = speed2 / speed1 if speed1 > 0 else 0
    
    # Perplexity comparison (lower is better)
    ppl1 = model1_results['perplexity']
    ppl2 = model2_results['perplexity']
    comparison['perplexity_change'] = ((ppl2 - ppl1) / ppl1 * 100) if ppl1 > 0 else 0
    
    # Model size comparison
    size1 = model1_results['model_size']['total_params']
    size2 = model2_results['model_size']['total_params']
    comparison['param_reduction'] = ((size1 - size2) / size1 * 100) if size1 > 0 else 0
    
    # Compression error (if applicable)
    if 'compression' in model2_results:
        comparison['reconstruction_error'] = model2_results['compression']['avg_reconstruction_error']
        comparison['compression_time'] = model2_results['compression']['total_compression_time']
    
    return comparison

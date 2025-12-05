"""
Experiment runner for automating model training and evaluation.

This module provides utilities for running experiments with different
configurations, collecting metrics, and saving results.
"""

import os
import json
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from mla_gpt.model import GPT, GPTConfig
from mla_gpt.utils.metrics import (
    compute_perplexity,
    measure_inference_speed,
    extract_compression_metrics_from_model,
    compute_reconstruction_error,
)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    
    # Model configuration
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 256
    dropout: float = 0.0
    bias: bool = False
    
    # MLA configuration
    use_mla: bool = False
    kv_latent_dim: Optional[int] = None
    q_latent_dim: Optional[int] = None
    
    # MLA KV Compression configuration
    kv_compression_type: str = 'none'  # 'none', 'svd', or 'randomized_svd'
    kv_compression_rank: Optional[int] = None
    
    # Randomized SVD parameters
    svd_n_oversamples: int = 10
    svd_n_power_iter: int = 2
    
    # Training configuration
    batch_size: int = 4
    max_iters: int = 100
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    
    # Evaluation configuration
    eval_iters: int = 10
    eval_interval: int = 50


@dataclass
class ExperimentResults:
    """Results from a single experiment."""
    config: ExperimentConfig
    
    # Model metrics
    model_params: int
    trainable_params: int
    
    # Quality metrics
    final_train_loss: float
    final_val_loss: float
    final_train_perplexity: float
    final_val_perplexity: float
    
    # Performance metrics
    training_tokens_per_sec: float
    inference_tokens_per_sec: float
    avg_training_time_per_iter: float
    total_training_time: float
    
    # Memory metrics
    forward_memory_mb: float
    backward_memory_mb: float
    total_memory_mb: float
    
    # Compression metrics (if applicable)
    compression_metrics: Optional[Dict[str, Any]] = None
    
    # Training history
    train_losses: List[float] = None
    val_losses: List[float] = None
    iteration_times: List[float] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['config'] = asdict(self.config)
        return result
    
    def save(self, filepath):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct config
        config = ExperimentConfig(**data['config'])
        data['config'] = config
        
        return cls(**data)


class ExperimentRunner:
    """
    Runs experiments with different configurations and collects metrics.
    """
    
    def __init__(self, data_dir='data', output_dir='experiments/results', device='cuda'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device if torch.cuda.is_available() else 'cpu'
        
    def create_model(self, config: ExperimentConfig) -> GPT:
        """Create a GPT model from experiment configuration."""
        model_config = GPTConfig(
            block_size=config.block_size,
            vocab_size=50304,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
            # MLA
            use_mla=config.use_mla,
            kv_latent_dim=config.kv_latent_dim,
            q_latent_dim=config.q_latent_dim,
            # MLA KV Compression
            kv_compression_type=config.kv_compression_type,
            kv_compression_rank=config.kv_compression_rank,
            # Randomized SVD parameters
            svd_n_oversamples=config.svd_n_oversamples,
            svd_n_power_iter=config.svd_n_power_iter,
        )
        
        model = GPT(model_config)
        model.to(self.device)
        return model
    
    def load_data(self, dataset_name='shakespeare_char', batch_size=4, block_size=256):
        """
        Load training and validation data.
        
        Args:
            dataset_name: Name of the dataset ('shakespeare_char', 'shakespeare', 'openwebtext')
            batch_size: Batch size
            block_size: Context length
            
        Returns:
            train_loader, val_loader
        """
        data_path = self.data_dir / dataset_name
        
        # Load the data files
        train_data = np.memmap(data_path / 'train.bin', dtype=np.uint16, mode='r')
        val_data = np.memmap(data_path / 'val.bin', dtype=np.uint16, mode='r')
        
        def get_batch(split):
            data = train_data if split == 'train' else val_data
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
            return x.to(self.device), y.to(self.device)
        
        return get_batch
    
    def train_model(self, model: GPT, get_batch, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Train a model and collect training metrics.
        
        Args:
            model: The GPT model to train
            get_batch: Function to get training batches
            config: Experiment configuration
            
        Returns:
            dict with training history and metrics
        """
        optimizer = model.configure_optimizers(
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=(0.9, 0.95),
            device_type=self.device
        )
        
        train_losses = []
        val_losses = []
        iteration_times = []
        
        model.train()
        total_start = time.time()
        
        for iter_num in range(config.max_iters):
            iter_start = time.time()
            
            # Get batch
            X, Y = get_batch('train')
            
            # Forward and backward
            logits, loss = model(X, Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            train_losses.append(loss.item())
            
            # Evaluate
            if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
                model.eval()
                val_loss_accum = 0.0
                with torch.no_grad():
                    for _ in range(config.eval_iters):
                        X_val, Y_val = get_batch('val')
                        _, val_loss = model(X_val, Y_val)
                        val_loss_accum += val_loss.item()
                val_loss_accum /= config.eval_iters
                val_losses.append(val_loss_accum)
                
                print(f"Iter {iter_num}/{config.max_iters}: "
                      f"train_loss={loss.item():.4f}, val_loss={val_loss_accum:.4f}")
                
                model.train()
        
        total_time = time.time() - total_start
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'iteration_times': iteration_times,
            'total_time': total_time,
            'avg_time_per_iter': np.mean(iteration_times),
        }
    
    def evaluate_model(self, model: GPT, get_batch, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Evaluate model and collect all metrics.
        
        Args:
            model: The trained GPT model
            get_batch: Function to get evaluation batches
            config: Experiment configuration
            
        Returns:
            dict with evaluation metrics
        """
        model.eval()
        
        # Compute perplexity on validation set
        val_loss_accum = 0.0
        with torch.no_grad():
            for _ in range(config.eval_iters):
                X, Y = get_batch('val')
                _, loss = model(X, Y)
                val_loss_accum += loss.item()
        val_loss = val_loss_accum / config.eval_iters
        val_perplexity = np.exp(val_loss)
        
        # Compute perplexity on training set
        train_loss_accum = 0.0
        with torch.no_grad():
            for _ in range(config.eval_iters):
                X, Y = get_batch('train')
                _, loss = model(X, Y)
                train_loss_accum += loss.item()
        train_loss = train_loss_accum / config.eval_iters
        train_perplexity = np.exp(train_loss)
        
        # Measure inference speed
        inference_times = []
        with torch.no_grad():
            for _ in range(10):
                X, _ = get_batch('val')
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.time()
                _ = model(X)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                inference_times.append(time.time() - start)
        
        avg_inference_time = np.mean(inference_times)
        tokens_per_batch = config.batch_size * config.block_size
        inference_tokens_per_sec = tokens_per_batch / avg_inference_time
        
        # Memory metrics (simplified - not our focus)
        memory_metrics = {
            'forward_memory_mb': 0.0,
            'backward_memory_mb': 0.0,
            'total_memory_mb': 0.0,
        }
        
        return {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_perplexity': train_perplexity,
            'val_perplexity': val_perplexity,
            'inference_tokens_per_sec': inference_tokens_per_sec,
            'avg_inference_time': avg_inference_time,
            **memory_metrics,
        }
    
    def evaluate_compression(self, model: GPT, config: ExperimentConfig) -> Optional[Dict[str, Any]]:
        """
        Evaluate compression quality for MLA models with KV compression.
        
        Args:
            model: The GPT model
            config: Experiment configuration
            
        Returns:
            dict with compression metrics or None
        """
        # Check if this is an MLA model with KV compression
        if not config.use_mla or config.kv_compression_type == 'none':
            return None
        
        # Extract compression metrics from MLA attention layers
        # Metrics are stored during model initialization/forward pass
        compression_metrics = extract_compression_metrics_from_model(model)
        return compression_metrics if compression_metrics['num_compressed_layers'] > 0 else None
    
    def run_experiment(self, config: ExperimentConfig, dataset='shakespeare_char') -> ExperimentResults:
        """
        Run a complete experiment with the given configuration.
        
        Args:
            config: Experiment configuration
            dataset: Dataset name
            
        Returns:
            ExperimentResults object
        """
        print(f"\n{'='*60}")
        print(f"Running experiment: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*60}\n")
        
        # Create model
        model = self.create_model(config)
        num_params = model.get_num_params()
        print(f"Model parameters: {num_params:,}")
        
        # Load data
        get_batch = self.load_data(dataset, config.batch_size, config.block_size)
        
        # Train model
        print("\nTraining model...")
        training_results = self.train_model(model, get_batch, config)
        
        # Evaluate model
        print("\nEvaluating model...")
        eval_results = self.evaluate_model(model, get_batch, config)
        
        # Evaluate compression (if applicable)
        compression_results = self.evaluate_compression(model, config)
        
        # Compute training throughput
        tokens_per_iter = config.batch_size * config.block_size
        training_tokens_per_sec = tokens_per_iter / training_results['avg_time_per_iter']
        
        # Create results object
        results = ExperimentResults(
            config=config,
            model_params=num_params,
            trainable_params=num_params,  # Assuming all params are trainable
            final_train_loss=training_results['train_losses'][-1],
            final_val_loss=training_results['val_losses'][-1],
            final_train_perplexity=eval_results['train_perplexity'],
            final_val_perplexity=eval_results['val_perplexity'],
            training_tokens_per_sec=training_tokens_per_sec,
            inference_tokens_per_sec=eval_results['inference_tokens_per_sec'],
            avg_training_time_per_iter=training_results['avg_time_per_iter'],
            total_training_time=training_results['total_time'],
            forward_memory_mb=eval_results['forward_memory_mb'],
            backward_memory_mb=eval_results['backward_memory_mb'],
            total_memory_mb=eval_results['total_memory_mb'],
            compression_metrics=compression_results,
            train_losses=training_results['train_losses'],
            val_losses=training_results['val_losses'],
            iteration_times=training_results['iteration_times'],
        )
        
        # Save results
        output_file = self.output_dir / f"{config.name}_results.json"
        results.save(output_file)
        print(f"\nResults saved to: {output_file}")
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: ExperimentResults):
        """Print a summary of experiment results."""
        print(f"\n{'='*60}")
        print(f"Experiment Summary: {results.config.name}")
        print(f"{'='*60}")
        print(f"Model Parameters: {results.model_params:,}")
        print(f"\nQuality Metrics:")
        print(f"  Train Loss: {results.final_train_loss:.4f}")
        print(f"  Val Loss: {results.final_val_loss:.4f}")
        print(f"  Train Perplexity: {results.final_train_perplexity:.2f}")
        print(f"  Val Perplexity: {results.final_val_perplexity:.2f}")
        print(f"\nPerformance Metrics:")
        print(f"  Training Speed: {results.training_tokens_per_sec:.0f} tokens/sec")
        print(f"  Inference Speed: {results.inference_tokens_per_sec:.0f} tokens/sec")
        print(f"  Avg Time per Iteration: {results.avg_training_time_per_iter:.4f} sec")
        print(f"\nMemory Usage:")
        print(f"  Forward Pass: {results.forward_memory_mb:.2f} MB")
        print(f"  Backward Pass: {results.backward_memory_mb:.2f} MB")
        print(f"  Total: {results.total_memory_mb:.2f} MB")
        
        if results.compression_metrics and results.compression_metrics.get('num_compressed_layers', 0) > 0:
            metrics = results.compression_metrics
            print(f"\nCompression Metrics:")
            print(f"  Compression Type: {metrics.get('compression_type', 'none')}")
            print(f"  Compression Rank: {metrics.get('compression_rank', 'N/A')}")
            print(f"  Compressed Layers: {metrics.get('num_compressed_layers', 0)}")
            print(f"  Avg Reconstruction Error: {metrics.get('avg_reconstruction_error', 0):.6f}")
            print(f"  Avg Compression Time: {metrics.get('avg_compression_time', 0):.4f} sec")
            print(f"  Total Compression Time: {metrics.get('total_compression_time', 0):.4f} sec")
        
        print(f"{'='*60}\n")
    
    def run_experiments(self, configs: List[ExperimentConfig], dataset='shakespeare_char') -> List[ExperimentResults]:
        """
        Run multiple experiments and collect results.
        
        Args:
            configs: List of experiment configurations
            dataset: Dataset name
            
        Returns:
            List of ExperimentResults
        """
        results = []
        
        for config in configs:
            try:
                result = self.run_experiment(config, dataset)
                results.append(result)
            except Exception as e:
                print(f"Error running experiment {config.name}: {e}")
                import traceback
                traceback.print_exc()
        
        return results

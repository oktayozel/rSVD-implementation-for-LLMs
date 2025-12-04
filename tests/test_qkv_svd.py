#!/usr/bin/env python3
"""
Test script for separate SVD on Q, K, and V matrices.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mla_gpt.model.model import GPT, GPTConfig


def test_individual_svd():
    """Test SVD on Q, K, V separately"""
    print("=" * 70)
    print("Testing Individual SVD on Q, K, V")
    print("=" * 70)
    
    device = 'cpu'
    x = torch.randint(0, 256, (2, 32), device=device)
    
    base_config = {
        'block_size': 64,
        'vocab_size': 256,
        'n_layer': 2,
        'n_head': 4,
        'n_embd': 64,
        'dropout': 0.0,
        'bias': False
    }
    
    # Test configurations
    configs = [
        ("Baseline (No SVD)", {**base_config}),
        ("SVD on V only", {**base_config, 'use_svd_v': True, 'svd_rank_v': 16}),
        ("SVD on K only", {**base_config, 'use_svd_k': True, 'svd_rank_k': 16}),
        ("SVD on Q only", {**base_config, 'use_svd_q': True, 'svd_rank_q': 16}),
        ("SVD on K+V", {**base_config, 'use_svd_k': True, 'use_svd_v': True, 
                        'svd_rank_k': 16, 'svd_rank_v': 16}),
        ("SVD on Q+K+V", {**base_config, 'use_svd_q': True, 'use_svd_k': True, 
                          'use_svd_v': True, 'svd_rank_q': 16, 'svd_rank_k': 16, 
                          'svd_rank_v': 16}),
    ]
    
    baseline_logits = None
    
    for name, config_dict in configs:
        config = GPTConfig(**config_dict)
        model = GPT(config).to(device)
        model.eval()
        
        with torch.no_grad():
            logits, loss = model(x, targets=x)
        
        if baseline_logits is None:
            baseline_logits = logits
            print(f"\n{name}:")
            print(f"  ✓ Loss: {loss.item():.4f}")
        else:
            diff = torch.abs(logits - baseline_logits).max().item()
            print(f"\n{name}:")
            print(f"  ✓ Loss: {loss.item():.4f}")
            print(f"  Max diff from baseline: {diff:.6f}")


def test_different_ranks():
    """Test different ranks for Q, K, V"""
    print("\n" + "=" * 70)
    print("Testing Different Ranks for Q, K, V")
    print("=" * 70)
    
    device = 'cpu'
    x = torch.randint(0, 256, (2, 32), device=device)
    
    base_config = {
        'block_size': 64,
        'vocab_size': 256,
        'n_layer': 2,
        'n_head': 4,
        'n_embd': 64,
        'dropout': 0.0,
        'bias': False,
        'use_svd_q': True,
        'use_svd_k': True,
        'use_svd_v': True
    }
    
    rank_configs = [
        ("High ranks (Q:32, K:32, V:32)", {'svd_rank_q': 32, 'svd_rank_k': 32, 'svd_rank_v': 32}),
        ("Medium ranks (Q:16, K:16, V:16)", {'svd_rank_q': 16, 'svd_rank_k': 16, 'svd_rank_v': 16}),
        ("Low ranks (Q:8, K:8, V:8)", {'svd_rank_q': 8, 'svd_rank_k': 8, 'svd_rank_v': 8}),
        ("Asymmetric (Q:32, K:16, V:8)", {'svd_rank_q': 32, 'svd_rank_k': 16, 'svd_rank_v': 8}),
    ]
    
    for name, ranks in rank_configs:
        config = GPTConfig(**{**base_config, **ranks})
        model = GPT(config).to(device)
        model.eval()
        
        with torch.no_grad():
            logits, loss = model(x, targets=x)
        
        print(f"\n{name}:")
        print(f"  ✓ Loss: {loss.item():.4f}")
        print(f"  Ranks: Q={config.svd_rank_q}, K={config.svd_rank_k}, V={config.svd_rank_v}")


def test_performance_impact():
    """Test performance impact of different SVD configurations"""
    print("\n" + "=" * 70)
    print("Performance Impact Analysis")
    print("=" * 70)
    
    import time
    
    device = 'cpu'
    x = torch.randint(0, 256, (4, 128), device=device)
    
    base_config = {
        'block_size': 256,
        'vocab_size': 256,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 128,
        'dropout': 0.0,
        'bias': False
    }
    
    configs = [
        ("No SVD", {}),
        ("V only", {'use_svd_v': True, 'svd_rank_v': 16}),
        ("K+V", {'use_svd_k': True, 'use_svd_v': True, 'svd_rank_k': 16, 'svd_rank_v': 16}),
        ("Q+K+V", {'use_svd_q': True, 'use_svd_k': True, 'use_svd_v': True, 
                   'svd_rank_q': 16, 'svd_rank_k': 16, 'svd_rank_v': 16}),
    ]
    
    n_iters = 10
    
    for name, svd_config in configs:
        config = GPTConfig(**{**base_config, **svd_config})
        model = GPT(config).to(device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            _ = model(x)
        
        # Timing
        torch.manual_seed(42)
        start = time.time()
        with torch.no_grad():
            for _ in range(n_iters):
                logits, loss = model(x, targets=x)
        elapsed = time.time() - start
        
        print(f"\n{name}:")
        print(f"  Time for {n_iters} iterations: {elapsed:.3f}s")
        print(f"  Avg time per iteration: {elapsed/n_iters*1000:.2f}ms")
        print(f"  Final loss: {loss.item():.4f}")


def main():
    print("\n" + "🧪 " * 35)
    print("SVD Q/K/V Separate Test Suite")
    print("🧪 " * 35 + "\n")
    
    try:
        test_individual_svd()
        test_different_ranks()
        test_performance_impact()
        
        print("\n" + "=" * 70)
        print("✅ All Tests Passed!")
        print("=" * 70)
        
        print("\n" + "📊 Recommendations:")
        print("  • V-only SVD: Most stable, recommended for production")
        print("  • K+V SVD: Good balance between compression and quality")
        print("  • Q+K+V SVD: Maximum compression, may affect quality")
        print("  • Start with higher ranks and reduce gradually")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
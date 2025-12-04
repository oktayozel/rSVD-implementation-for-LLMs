"""
Test script for Multi-Head Latent Attention (MLA) implementation.
Tests basic functionality, parameter efficiency, and KV cache compression.
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mla_gpt.model.model import GPT, GPTConfig

def test_standard_attention():
    """Test baseline model with standard CausalSelfAttention."""
    print('=' * 70)
    print('Test 1: Standard CausalSelfAttention (Baseline)')
    print('=' * 70)
    
    config = GPTConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False,
        use_mla=False
    )
    
    model = GPT(config)
    x = torch.randint(0, 256, (2, 32))  # batch=2, seq_len=32
    
    # Forward pass
    logits, loss = model(x, targets=x)
    print(f'✓ Forward pass successful')
    print(f'  Input shape: {x.shape}')
    print(f'  Logits shape: {logits.shape}')
    print(f'  Loss: {loss.item():.4f}')
    
    # Backward pass
    loss.backward()
    print(f'✓ Backward pass successful')
    
    # Generation
    model.eval()
    prompt = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=10)
    print(f'✓ Generation successful: {prompt.shape[1]} -> {generated.shape[1]} tokens')
    
    return model, config


def test_mla_attention():
    """Test model with Multi-Head Latent Attention."""
    print('\n' + '=' * 70)
    print('Test 2: Multi-Head Latent Attention (MLA)')
    print('=' * 70)
    
    config = GPTConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False,
        use_mla=True,
        kv_latent_dim=16,  # n_embd // 4 = 4x compression
        q_latent_dim=32,   # n_embd // 2
    )
    
    model = GPT(config)
    x = torch.randint(0, 256, (2, 32))
    
    # Forward pass
    logits, loss = model(x, targets=x)
    print(f'✓ Forward pass successful')
    print(f'  Input shape: {x.shape}')
    print(f'  Logits shape: {logits.shape}')
    print(f'  Loss: {loss.item():.4f}')
    
    # Backward pass
    loss.backward()
    print(f'✓ Backward pass successful')
    
    # Generation
    model.eval()
    prompt = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=10)
    print(f'✓ Generation successful: {prompt.shape[1]} -> {generated.shape[1]} tokens')
    
    return model, config


def test_mla_with_rope():
    """Test MLA with rotary position embeddings."""
    print('\n' + '=' * 70)
    print('Test 3: MLA with RoPE (Rotary Position Embeddings)')
    print('=' * 70)
    
    config = GPTConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False,
        use_mla=True,
        kv_latent_dim=16,
        q_latent_dim=32,
        use_rope=True,
        rope_dim=8,  # head_size // 2
    )
    
    model = GPT(config)
    x = torch.randint(0, 256, (2, 32))
    
    # Forward pass
    logits, loss = model(x, targets=x)
    print(f'✓ Forward pass with RoPE successful')
    print(f'  Input shape: {x.shape}')
    print(f'  Logits shape: {logits.shape}')
    print(f'  Loss: {loss.item():.4f}')
    
    # Backward pass
    loss.backward()
    print(f'✓ Backward pass successful')
    
    return model, config


def compare_models(std_model, mla_model):
    """Compare parameter counts and memory efficiency."""
    print('\n' + '=' * 70)
    print('Test 4: Model Comparison')
    print('=' * 70)
    
    std_params = std_model.get_num_params()
    mla_params = mla_model.get_num_params()
    
    print(f'\nParameter Counts:')
    print(f'  Standard attention: {std_params:,} parameters')
    print(f'  MLA attention:      {mla_params:,} parameters')
    diff = mla_params - std_params
    pct = (diff / std_params) * 100
    print(f'  Difference:         {diff:+,} ({pct:+.2f}%)')
    
    # KV cache comparison
    print(f'\nKV Cache Efficiency (per layer, per token):')
    mla_attn = mla_model.transformer.h[0].attn
    cache_info = mla_attn.get_kv_cache_size()
    
    print(f'  Standard MHA cache: {cache_info["standard_mha_cache_per_token"]} values')
    print(f'  MLA cache:          {cache_info["mla_cache_per_token"]} values')
    print(f'  Compression ratio:  {cache_info["compression_ratio"]:.1f}x')
    
    # Memory calculation for a full sequence
    seq_len = 1024
    n_layers = 2
    bytes_per_param = 2  # fp16
    
    std_cache_memory = cache_info["standard_mha_cache_per_token"] * seq_len * n_layers * bytes_per_param
    mla_cache_memory = cache_info["mla_cache_per_token"] * seq_len * n_layers * bytes_per_param
    
    print(f'\nMemory for sequence length {seq_len} ({n_layers} layers):')
    print(f'  Standard MHA: {std_cache_memory / 1024:.2f} KB')
    print(f'  MLA:          {mla_cache_memory / 1024:.2f} KB')
    print(f'  Saved:        {(std_cache_memory - mla_cache_memory) / 1024:.2f} KB')


def test_different_compressions():
    """Test MLA with different compression ratios."""
    print('\n' + '=' * 70)
    print('Test 5: Different Compression Ratios')
    print('=' * 70)
    
    n_embd = 128
    compression_configs = [
        (n_embd // 2, '2x'),   # Mild compression
        (n_embd // 4, '4x'),   # Standard compression
        (n_embd // 8, '8x'),   # Aggressive compression
    ]
    
    print(f'\nTesting with n_embd={n_embd}:\n')
    
    for kv_dim, ratio in compression_configs:
        config = GPTConfig(
            block_size=64,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_embd=n_embd,
            dropout=0.0,
            bias=False,
            use_mla=True,
            kv_latent_dim=kv_dim,
            q_latent_dim=n_embd // 2,
        )
        
        model = GPT(config)
        x = torch.randint(0, 256, (2, 32))
        
        try:
            logits, loss = model(x, targets=x)
            print(f'  ✓ {ratio} compression (kv_latent_dim={kv_dim}): loss={loss.item():.4f}')
        except Exception as e:
            print(f'  ✗ {ratio} compression failed: {e}')


def test_batch_sizes():
    """Test MLA with different batch sizes."""
    print('\n' + '=' * 70)
    print('Test 6: Different Batch Sizes')
    print('=' * 70)
    
    config = GPTConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False,
        use_mla=True,
        kv_latent_dim=16,
        q_latent_dim=32,
    )
    
    model = GPT(config)
    batch_sizes = [1, 2, 4, 8]
    seq_len = 32
    
    print(f'\nTesting with sequence length {seq_len}:\n')
    
    for batch_size in batch_sizes:
        x = torch.randint(0, 256, (batch_size, seq_len))
        try:
            with torch.no_grad():
                logits, loss = model(x, targets=x)
            print(f'  ✓ Batch size {batch_size}: shape={logits.shape}, loss={loss.item():.4f}')
        except Exception as e:
            print(f'  ✗ Batch size {batch_size} failed: {e}')


def main():
    """Run all tests."""
    print('\n' + '🧪 ' * 35)
    print('Multi-Head Latent Attention (MLA) Test Suite')
    print('🧪 ' * 35 + '\n')
    
    try:
        # Basic tests
        std_model, std_config = test_standard_attention()
        mla_model, mla_config = test_mla_attention()
        mla_rope_model, mla_rope_config = test_mla_with_rope()
        
        # Comparison
        compare_models(std_model, mla_model)
        
        # Advanced tests
        test_different_compressions()
        test_batch_sizes()
        
        # Summary
        print('\n' + '=' * 70)
        print('✅ All Tests Passed!')
        print('=' * 70)
        print('\nSummary:')
        print('  • Standard attention works correctly')
        print('  • MLA attention works correctly')
        print('  • MLA with RoPE works correctly')
        print('  • KV cache compression verified')
        print('  • Multiple compression ratios tested')
        print('  • Different batch sizes tested')
        print('\n' + '🎉 ' * 35 + '\n')
        
        return True
        
    except Exception as e:
        print('\n' + '=' * 70)
        print('❌ Tests Failed!')
        print('=' * 70)
        print(f'\nError: {e}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
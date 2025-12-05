"""
MLA Experiment Configurations with SVD/rSVD Compression on KV Projections.

This file defines experiments comparing Multi-Head Latent Attention (MLA) with:
- No compression (baseline MLA)
- Standard SVD compression on KV down-projection (various ranks)
- Randomized SVD compression on KV down-projection (various ranks)

All experiments use MLA attention. The only variable is the compression applied
to the KV down-projection matrix.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mla_gpt.utils.experiment_runner import ExperimentConfig


# =============================================================================
# Helper Functions for Creating MLA Configs
# =============================================================================

def create_mla_config(name, description, n_layer, n_head, n_embd, kv_latent_dim,
                      kv_compression_type='none', kv_compression_rank=None,
                      max_iters=1000, batch_size=4, learning_rate=6e-4):
    """
    Create an MLA experiment configuration.
    
    Args:
        name: Experiment name
        description: Experiment description
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        kv_latent_dim: KV latent space dimension (controls KV cache compression)
        kv_compression_type: 'none', 'svd', or 'randomized_svd'
        kv_compression_rank: Rank for SVD compression (None for no compression)
        max_iters: Training iterations
        batch_size: Batch size
        learning_rate: Learning rate
    """
    config = ExperimentConfig(
        name=name,
        description=description,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=256,
        batch_size=batch_size,
        max_iters=max_iters,
        eval_iters=50,
        eval_interval=100,
        learning_rate=learning_rate,
        # MLA configuration
        use_mla=True,
        kv_latent_dim=kv_latent_dim,
        q_latent_dim=n_embd // 2,  # 2x compression for queries
        # KV compression configuration
        kv_compression_type=kv_compression_type,
        kv_compression_rank=kv_compression_rank,
        # Randomized SVD parameters
        svd_n_oversamples=10,
        svd_n_power_iter=2,
    )
    return config


# =============================================================================
# Model Size Presets
# =============================================================================

# Small model preset (fast experiments)
SMALL_MODEL = {
    'n_layer': 4,
    'n_head': 4,
    'n_embd': 128,
    'kv_latent_dim': 32,  # 4x compression
}

# Medium model preset
MEDIUM_MODEL = {
    'n_layer': 6,
    'n_head': 6,
    'n_embd': 192,
    'kv_latent_dim': 48,  # 4x compression
}

# Large model preset
LARGE_MODEL = {
    'n_layer': 8,
    'n_head': 8,
    'n_embd': 256,
    'kv_latent_dim': 64,  # 4x compression
}


# =============================================================================
# Baseline MLA Experiments (No KV Compression)
# =============================================================================

MLA_BASELINE_SMALL = create_mla_config(
    name='mla_baseline_small',
    description='Small MLA without KV compression (baseline)',
    **SMALL_MODEL,
    kv_compression_type='none',
)

MLA_BASELINE_MEDIUM = create_mla_config(
    name='mla_baseline_medium',
    description='Medium MLA without KV compression (baseline)',
    **MEDIUM_MODEL,
    kv_compression_type='none',
)

MLA_BASELINE_LARGE = create_mla_config(
    name='mla_baseline_large',
    description='Large MLA without KV compression (baseline)',
    **LARGE_MODEL,
    kv_compression_type='none',
)


# =============================================================================
# Standard SVD Compression Experiments
# =============================================================================

MLA_SVD_RANK8 = create_mla_config(
    name='mla_svd_rank8',
    description='MLA with standard SVD compression (rank 8) on KV projection',
    **SMALL_MODEL,
    kv_compression_type='svd',
    kv_compression_rank=8,
)

MLA_SVD_RANK16 = create_mla_config(
    name='mla_svd_rank16',
    description='MLA with standard SVD compression (rank 16) on KV projection',
    **SMALL_MODEL,
    kv_compression_type='svd',
    kv_compression_rank=16,
)

MLA_SVD_RANK24 = create_mla_config(
    name='mla_svd_rank24',
    description='MLA with standard SVD compression (rank 24) on KV projection',
    **SMALL_MODEL,
    kv_compression_type='svd',
    kv_compression_rank=24,
)

MLA_SVD_RANK32 = create_mla_config(
    name='mla_svd_rank32',
    description='MLA with standard SVD compression (rank 32) on KV projection',
    **SMALL_MODEL,
    kv_compression_type='svd',
    kv_compression_rank=32,
)


# =============================================================================
# Randomized SVD Compression Experiments
# =============================================================================

MLA_RSVD_RANK8 = create_mla_config(
    name='mla_rsvd_rank8',
    description='MLA with randomized SVD compression (rank 8) on KV projection',
    **SMALL_MODEL,
    kv_compression_type='randomized_svd',
    kv_compression_rank=8,
)

MLA_RSVD_RANK16 = create_mla_config(
    name='mla_rsvd_rank16',
    description='MLA with randomized SVD compression (rank 16) on KV projection',
    **SMALL_MODEL,
    kv_compression_type='randomized_svd',
    kv_compression_rank=16,
)

MLA_RSVD_RANK24 = create_mla_config(
    name='mla_rsvd_rank24',
    description='MLA with randomized SVD compression (rank 24) on KV projection',
    **SMALL_MODEL,
    kv_compression_type='randomized_svd',
    kv_compression_rank=24,
)

MLA_RSVD_RANK32 = create_mla_config(
    name='mla_rsvd_rank32',
    description='MLA with randomized SVD compression (rank 32) on KV projection',
    **SMALL_MODEL,
    kv_compression_type='randomized_svd',
    kv_compression_rank=32,
)


# =============================================================================
# Medium Model Variants
# =============================================================================

MLA_MEDIUM_SVD_RANK16 = create_mla_config(
    name='mla_medium_svd_rank16',
    description='Medium MLA with standard SVD compression (rank 16)',
    **MEDIUM_MODEL,
    kv_compression_type='svd',
    kv_compression_rank=16,
)

MLA_MEDIUM_RSVD_RANK16 = create_mla_config(
    name='mla_medium_rsvd_rank16',
    description='Medium MLA with randomized SVD compression (rank 16)',
    **MEDIUM_MODEL,
    kv_compression_type='randomized_svd',
    kv_compression_rank=16,
)


# =============================================================================
# Large Model Variants
# =============================================================================

MLA_LARGE_SVD_RANK32 = create_mla_config(
    name='mla_large_svd_rank32',
    description='Large MLA with standard SVD compression (rank 32)',
    **LARGE_MODEL,
    kv_compression_type='svd',
    kv_compression_rank=32,
)

MLA_LARGE_RSVD_RANK32 = create_mla_config(
    name='mla_large_rsvd_rank32',
    description='Large MLA with randomized SVD compression (rank 32)',
    **LARGE_MODEL,
    kv_compression_type='randomized_svd',
    kv_compression_rank=32,
)


# =============================================================================
# Predefined Experiment Suites
# =============================================================================

# Quick test suite (3 experiments, ~10 minutes)
QUICK_TEST_SUITE = [
    MLA_BASELINE_SMALL,
    MLA_SVD_RANK16,
    MLA_RSVD_RANK16,
]

# Compression comparison suite (baseline + all ranks)
COMPRESSION_COMPARISON_SUITE = [
    MLA_BASELINE_SMALL,
    MLA_SVD_RANK8,
    MLA_SVD_RANK16,
    MLA_SVD_RANK24,
    MLA_SVD_RANK32,
    MLA_RSVD_RANK8,
    MLA_RSVD_RANK16,
    MLA_RSVD_RANK24,
    MLA_RSVD_RANK32,
]

# SVD vs rSVD comparison (same ranks)
SVD_VS_RSVD_SUITE = [
    MLA_BASELINE_SMALL,
    MLA_SVD_RANK8,
    MLA_RSVD_RANK8,
    MLA_SVD_RANK16,
    MLA_RSVD_RANK16,
    MLA_SVD_RANK32,
    MLA_RSVD_RANK32,
]

# Model size ablation (different model sizes, same compression)
MODEL_SIZE_SUITE = [
    MLA_BASELINE_SMALL,
    MLA_BASELINE_MEDIUM,
    MLA_BASELINE_LARGE,
    MLA_SVD_RANK16,
    MLA_MEDIUM_SVD_RANK16,
    MLA_LARGE_SVD_RANK32,
    MLA_RSVD_RANK16,
    MLA_MEDIUM_RSVD_RANK16,
    MLA_LARGE_RSVD_RANK32,
]

# Rank ablation (test different ranks)
RANK_ABLATION_SUITE = [
    MLA_BASELINE_SMALL,
    MLA_RSVD_RANK8,
    MLA_RSVD_RANK16,
    MLA_RSVD_RANK24,
    MLA_RSVD_RANK32,
]

# Comprehensive suite (all experiments)
COMPREHENSIVE_SUITE = [
    # Baselines
    MLA_BASELINE_SMALL,
    MLA_BASELINE_MEDIUM,
    MLA_BASELINE_LARGE,
    # Small model variants
    MLA_SVD_RANK8,
    MLA_SVD_RANK16,
    MLA_SVD_RANK24,
    MLA_SVD_RANK32,
    MLA_RSVD_RANK8,
    MLA_RSVD_RANK16,
    MLA_RSVD_RANK24,
    MLA_RSVD_RANK32,
    # Medium model variants
    MLA_MEDIUM_SVD_RANK16,
    MLA_MEDIUM_RSVD_RANK16,
    # Large model variants
    MLA_LARGE_SVD_RANK32,
    MLA_LARGE_RSVD_RANK32,
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_suite(suite_name: str):
    """
    Get a predefined experiment suite by name.
    
    Args:
        suite_name: Name of the suite
    
    Returns:
        List of ExperimentConfig objects
    """
    suites = {
        'quick': QUICK_TEST_SUITE,
        'compression': COMPRESSION_COMPARISON_SUITE,
        'svd_vs_rsvd': SVD_VS_RSVD_SUITE,
        'model_size': MODEL_SIZE_SUITE,
        'rank_ablation': RANK_ABLATION_SUITE,
        'comprehensive': COMPREHENSIVE_SUITE,
    }
    
    if suite_name not in suites:
        raise ValueError(f"Unknown suite: {suite_name}. Available: {list(suites.keys())}")
    
    return suites[suite_name]


def list_suites():
    """List all available experiment suites with descriptions."""
    suites = {
        'quick': {
            'description': 'Quick test (3 experiments: baseline, SVD, rSVD)',
            'experiments': 3,
            'time': '~10 min',
        },
        'compression': {
            'description': 'Full compression comparison (all ranks: 8, 16, 24, 32)',
            'experiments': 9,
            'time': '~30 min',
        },
        'svd_vs_rsvd': {
            'description': 'Direct SVD vs rSVD comparison (ranks: 8, 16, 32)',
            'experiments': 7,
            'time': '~20 min',
        },
        'model_size': {
            'description': 'Test different model sizes (small, medium, large)',
            'experiments': 9,
            'time': '~45 min',
        },
        'rank_ablation': {
            'description': 'Test different compression ranks (8, 16, 24, 32)',
            'experiments': 5,
            'time': '~15 min',
        },
        'comprehensive': {
            'description': 'All experiments (all models, all compression variants)',
            'experiments': 16,
            'time': '~60 min',
        },
    }
    
    print("\nAvailable MLA Experiment Suites")
    print("=" * 80)
    print(f"{'Suite Name':<20} {'Experiments':<12} {'Time':<12} {'Description'}")
    print("-" * 80)
    for name, info in suites.items():
        print(f"{name:<20} {info['experiments']:<12} {info['time']:<12} {info['description']}")
    print("=" * 80)
    print("\nUsage: python run_experiments.py --suite <suite_name>")
    print()


def list_all_experiments():
    """List all individual experiment configurations."""
    experiments = [
        ('Baselines', [MLA_BASELINE_SMALL, MLA_BASELINE_MEDIUM, MLA_BASELINE_LARGE]),
        ('Standard SVD (Small)', [MLA_SVD_RANK8, MLA_SVD_RANK16, MLA_SVD_RANK24, MLA_SVD_RANK32]),
        ('Randomized SVD (Small)', [MLA_RSVD_RANK8, MLA_RSVD_RANK16, MLA_RSVD_RANK24, MLA_RSVD_RANK32]),
        ('Medium Model', [MLA_MEDIUM_SVD_RANK16, MLA_MEDIUM_RSVD_RANK16]),
        ('Large Model', [MLA_LARGE_SVD_RANK32, MLA_LARGE_RSVD_RANK32]),
    ]
    
    print("\nAll Available Experiment Configurations")
    print("=" * 80)
    for category, configs in experiments:
        print(f"\n{category}:")
        for config in configs:
            print(f"  - {config.name:30s} : {config.description}")
    print("=" * 80)
    print()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--list-all':
        list_all_experiments()
    else:
        list_suites()

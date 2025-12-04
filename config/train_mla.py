# Configuration for GPT-2 training with Multi-Head Latent Attention (MLA)
# MLA reduces KV cache memory by compressing keys and values into a shared latent space
#
# Launch with:
# $ python train.py config/train_mla.py

wandb_log = False  # Set to True for experiment tracking
wandb_project = 'mla-experiments'
wandb_run_name = 'gpt2-mla-test'

# Batch configuration - smaller scale for testing
batch_size = 4
block_size = 256
gradient_accumulation_steps = 1

# Training iterations
max_iters = 1000
lr_decay_iters = 1000

# Evaluation settings
eval_interval = 100
eval_iters = 50
log_interval = 10

# Optimizer hyperparameters
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# Model architecture
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# ============ MLA Configuration ============
# Enable Multi-Head Latent Attention instead of standard attention
use_mla = True

# KV latent dimension - controls KV cache compression ratio
# Default: n_embd // 4 = 32 for n_embd=128
# Compression ratio: 2 * n_embd / kv_latent_dim = 8x with default
kv_latent_dim = 32

# Query latent dimension - controls query compression
# Default: n_embd // 2 = 64 for n_embd=128
# Set to n_embd to disable query compression
q_latent_dim = 64

# Rotary Position Embeddings (optional, as in DeepSeek-V2)
use_rope = False  # Set to True to enable decoupled RoPE
rope_dim = 16     # Dimension for RoPE portion of head (head_size // 2)

# ============ SVD Configuration ============
# SVD is separate from MLA - can be used together or independently
use_svd = False   # Disable SVD (MLA provides its own compression)
svd_rank = None

# Dataset
dataset = 'shakespeare_char'
# Configuration for testing separate SVD on Q, K, and V matrices

wandb_log = False
wandb_project = 'svd-qkv-experiments'
wandb_run_name = 'gpt2-svd-qkv-test'

# Training settings
batch_size = 4
block_size = 256
gradient_accumulation_steps = 1

max_iters = 1000
lr_decay_iters = 1000

eval_interval = 100
eval_iters = 50
log_interval = 10

# Optimizer
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

# ============ SVD Configuration - Separate Q, K, V ============

# Option 1: Apply SVD only to V (most robust, recommended)
use_svd_q = False
use_svd_k = False
use_svd_v = True
svd_rank_v = 16

# Option 2: Apply SVD to all Q, K, V (most aggressive compression)
# use_svd_q = True
# use_svd_k = True
# use_svd_v = True
# svd_rank_q = 32
# svd_rank_k = 32
# svd_rank_v = 16

# Option 3: Apply SVD to K and V only (balance between compression and quality)
# use_svd_q = False
# use_svd_k = True
# use_svd_v = True
# svd_rank_k = 24
# svd_rank_v = 16

# Dataset
dataset = 'shakespeare_char'
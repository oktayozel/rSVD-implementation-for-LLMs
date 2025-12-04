# Configuration for training with Randomized SVD

wandb_log = False
wandb_project = 'randomized-svd-experiments'
wandb_run_name = 'gpt2-rsvd-test'

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

# ============ Randomized SVD Configuration ============

# Enable randomized SVD
use_svd_q = True
use_svd_k = True
use_svd_v = True

# Ranks
svd_rank_q = 32
svd_rank_k = 24
svd_rank_v = 16

# Use randomized SVD
svd_type = 'randomized'  # 'standard' or 'randomized'

# Randomized SVD parameters
svd_n_oversamples = 10  # Additional samples for accuracy
svd_n_power_iter = 2     # Power iterations (0-4, higher = more accurate but slower)

# Dataset
dataset = 'shakespeare_char'
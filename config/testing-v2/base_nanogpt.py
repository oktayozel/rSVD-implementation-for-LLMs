# CPU Baseline - no compression

out_dir = 'out-wikitext103-cpu-base'

eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = False

dataset = 'wikitext-103'

# Small batches for CPU
batch_size = 4
gradient_accumulation_steps = 16  # Effective batch = 64
block_size = 512

# Medium model - CPU can handle this
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

# No compression
use_mla = False
use_svd_q = False
use_svd_k = False
use_svd_v = False

learning_rate = 3e-4
max_iters = 500  # Short demo run
lr_decay_iters = 500
min_lr = 3e-5
warmup_iters = 50

device = 'cpu'
compile = False  # PyTorch compile doesn't help much on CPU

# CPU-specific settings
dtype = 'bfloat16'  # Faster on modern CPUs
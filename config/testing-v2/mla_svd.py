# CPU with MLA + Standard SVD (SLOW)

out_dir = 'out-wikitext103-cpu-mla-svd'

eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = False

dataset = 'wikitext-103'

batch_size = 4
gradient_accumulation_steps = 16
block_size = 512

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

# Enable MLA
use_mla = True
kv_latent_dim = 192  # 768 // 4
q_latent_dim = 384   # 768 // 2
use_rope = False

# STANDARD SVD - will be VERY slow on CPU
kv_compression_type = 'svd'
kv_compression_rank = 64  # Aggressive: 192 → 64

learning_rate = 3e-4
max_iters = 500
lr_decay_iters = 500
min_lr = 3e-5
warmup_iters = 50

device = 'cpu'
compile = False
dtype = 'bfloat16'
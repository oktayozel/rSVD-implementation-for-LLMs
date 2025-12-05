# Medium Standard Attention + Randomized SVD on Q, K, V - ~124M params
out_dir = 'out-shakespeare-char-rsvd'

eval_interval = 500
eval_iters = 200
log_interval = 100
always_save_checkpoint = False

dataset = 'wikitext-103'

batch_size = 32
gradient_accumulation_steps = 8
block_size = 512

# Medium model config (~124M params)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

# Use standard attention (NOT MLA)
use_mla = False

# Enable randomized SVD on Q, K, V
use_svd_q = True
use_svd_k = True
use_svd_v = True
svd_rank_q = 48  # ~75% of head_dim (64)
svd_rank_k = 40  # ~62% of head_dim
svd_rank_v = 32  # ~50% of head_dim
svd_type = 'randomized'
svd_n_oversamples = 10
svd_n_power_iter = 2

learning_rate = 6e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 6e-5

warmup_iters = 100

device = 'cuda'
compile = True
# MLA with Standard SVD on Q, K, V
out_dir = 'out-shakespeare-char-mla-svd'

eval_interval = 500
eval_iters = 200
log_interval = 100
always_save_checkpoint = False

dataset = 'shakespeare_char'
batch_size = 64
gradient_accumulation_steps = 8
block_size = 256 

n_layer = 8
n_head = 12
n_embd = 384
dropout = 0.5

# Enable MLA with compression
use_mla = True
kv_latent_dim = 96   # n_embd // 4 = 4x KV cache compression
q_latent_dim = 192   # n_embd // 2 = 2x query compression
use_rope = False     # Disable RoPE for simpler comparison

# Enable standard SVD on Q, K, V within attention
use_svd_q = True
use_svd_k = True
use_svd_v = True
svd_rank_q = 24  # ~75% of head_dim (32)
svd_rank_k = 20  # ~62% of head_dim
svd_rank_v = 16  # ~50% of head_dim
svd_type = 'standard'

learning_rate = 3e-4
max_iters = 3000
lr_decay_iters = 3000
min_lr = 3e-5

warmup_iters = 100

device = 'cuda'
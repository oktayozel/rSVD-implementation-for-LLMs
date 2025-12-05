# Medium MLA model (MLA's built-in latent compression) - ~124M params
out_dir = 'out-shakespeare-char-mla-medium'

eval_interval = 500
eval_iters = 200
log_interval = 100
always_save_checkpoint = False

dataset = 'shakespeare_char'
batch_size = 32
gradient_accumulation_steps = 8
block_size = 512

# Medium model config (~124M params)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

# Enable MLA with latent compression
use_mla = True
kv_latent_dim = 192  # n_embd // 4 = 4x KV cache compression
q_latent_dim = 384   # n_embd // 2 = 2x query compression
use_rope = False

# No SVD - MLA provides its own compression
use_svd_q = False
use_svd_k = False
use_svd_v = False

learning_rate = 6e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 6e-5

warmup_iters = 100

device = 'cuda'
compile = True
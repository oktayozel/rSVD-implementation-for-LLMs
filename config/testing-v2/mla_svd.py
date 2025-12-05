# Large MLA model with Standard SVD compression - ~350M params
# MLA provides KV cache compression, SVD on KV projection

out_dir = 'out-wikitext103-mla-svd'

eval_interval = 1000
eval_iters = 200
log_interval = 50
always_save_checkpoint = False

dataset = 'wikitext-103'

batch_size = 16
gradient_accumulation_steps = 4
block_size = 1024

# Large model config (~350M params)
n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.1

# Enable MLA with latent compression
use_mla = True
kv_latent_dim = 256   # n_embd // 4 = 4x KV cache compression
q_latent_dim = 512    # n_embd // 2 = 2x query compression
use_rope = False

# Apply STANDARD SVD to KV down-projection for additional compression
# This will be SLOWER than rSVD but serves as comparison
kv_compression_type = 'svd'
kv_compression_rank = 128  # 50% compression of kv_latent_dim

# No separate Q/K/V SVD (MLA handles attention differently)
use_svd_q = False
use_svd_k = False
use_svd_v = False

learning_rate = 3e-4
max_iters = 10000
lr_decay_iters = 10000
min_lr = 3e-5

warmup_iters = 500

device = 'cuda'
compile = True
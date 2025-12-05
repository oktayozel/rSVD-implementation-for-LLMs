# Large MLA model with Randomized SVD compression - ~350M params
# Demonstrates rSVD speed advantage with minimal accuracy loss

out_dir = 'out-wikitext103-mla-rsvd'

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

# Apply RANDOMIZED SVD to KV down-projection
# This will be FASTER than standard SVD with similar quality
kv_compression_type = 'randomized_svd'
kv_compression_rank = 128  # Same rank as standard SVD for fair comparison

# Randomized SVD parameters - tuned for speed/accuracy balance
svd_n_oversamples = 10     # Additional samples for accuracy
svd_n_power_iter = 2       # Power iterations (2 is good balance)

# No separate Q/K/V SVD
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
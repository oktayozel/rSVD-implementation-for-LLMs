# Base Karpathy GPT model (no MLA, no SVD)
out_dir = 'out-shakespeare-char-base'

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

# No MLA, no SVD - standard attention
use_mla = False
use_svd_q = False
use_svd_k = False
use_svd_v = False

learning_rate = 3e-4
max_iters = 3000
lr_decay_iters = 3000
min_lr = 3e-5

warmup_iters = 100

device = 'cuda'
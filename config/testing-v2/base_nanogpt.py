# Large Base model (no MLA, no SVD) - ~350M params
# Baseline for comparison on WikiText-103

out_dir = 'out-wikitext103-base'

eval_interval = 1000
eval_iters = 200
log_interval = 50
always_save_checkpoint = False

dataset = 'wikitext-103'

batch_size = 16
gradient_accumulation_steps = 4
block_size = 1024

# Large model config (~350M params, GPT-2 Medium scale)
n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.1

# No compression - standard attention
use_mla = False
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
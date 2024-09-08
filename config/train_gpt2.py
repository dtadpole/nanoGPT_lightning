import argparse

# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

args.wandb_log = True
args.wandb_project = 'gpt2'
args.wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 16 batch size * 1024 block size * 5 gradaccum * 2 GPUs = 163,840 [491,520]
args.batch_size = 8
args.block_size = 1024
args.gradient_accumulation_steps = 40

# this makes total number of tokens be 300B
args.max_iters = 600000
args.lr_decay_iters = 600000

# eval stuff
args.eval_interval = 500
args.eval_iters = 400
args.log_interval = 20

# weight decay
args.weight_decay = 1e-1

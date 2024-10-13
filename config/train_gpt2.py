import argparse

# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

args.wandb_log = True
args.wandb_project = 'gpt2'
args.wandb_run_name = args.arch + '-124M'

# these make the total batch size be ~0.5M
# 10 batch size * 1024 block size * 32 gradaccum = 163,840 [327,680]
args.batch_size = 10
args.block_size = 1024
args.gradient_accumulation_steps = 32

# this makes total number of tokens be 300B
args.max_iters = 600000 // 4
args.lr_decay_iters = 600000 // 4
args.warmup_iters = 2000

# eval stuff
args.eval_interval = 2000 // 5
args.eval_iters = 400
args.log_interval = 20

# weight decay
args.weight_decay = 1e-1

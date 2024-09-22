import argparse
import os
import random
import inspect
import shutil
import time
import math
import numpy
import warnings
from enum import Enum
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LinearLR, ChainedScheduler, CosineAnnealingLR, CosineAnnealingWarmRestarts, SequentialLR
from torch.utils.data import Subset
import lightning as L
from lightning.fabric.accelerators import find_usable_cuda_devices
from gpt2 import GPT2, GPTConfig
from mamba import MyMamba, MyMambaConfig
# from transformers import MambaConfig, Mamba, MambaModel, MambaForCausalLM

# parser
parser = argparse.ArgumentParser(description='PyTorch GPT2 Training')
parser.add_argument('-a', '--arch', default='gpt2', type=str, metavar='ARCH',
                    help='model architecture (default: gpt2)')
parser.add_argument('-d', '--data', metavar='DIR', nargs='?', default='./data/openwebtext/',
                    help='path to dataset (default: ./data/openwebtext/)')
parser.add_argument('--config-file', default='./config/train_gpt2.py', type=str, metavar='PATH',
                    help='path to config file (default: ./config/train_gpt2.py)')
# batch and block size
parser.add_argument('-b', '--batch-size', default=12, type=int,
                    metavar='N',
                    help='mini-batch size (default: 12), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--block-size', default=1024, type=int,
                    help='block size (default: 1024)')
# learning rate
parser.add_argument('--lr', '--learning-rate', default=6e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='learning_rate')
parser.add_argument('--min-lr', default=6e-5, type=float,
                    metavar='min_lr', help='mean learning rate')
parser.add_argument('--beta1', default=0.9, type=float, metavar='B1',
                    help='beta1')
parser.add_argument('--beta2', default=0.95, type=float, metavar='B2',
                    help='beta2')
parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                    metavar='W', help='weight decay (default: 0.1)',
                    dest='weight_decay')
parser.add_argument('--clip-gradients', default=1.0, type=float,
                    metavar='CLIP_GRADIENTS', help='clip gradients')
parser.add_argument('--gradient-accumulation-steps', default=40, type=int,
                    metavar='GRADIENT_ACCUMULATION_STEPS', help='gradient accumulation steps')
# iterations
parser.add_argument('--warmup-iters', default=2000, type=int,
                    help='warmup epoch (default: 2000)')
parser.add_argument('--lr-decay-iters', default=600000, type=int,
                    metavar='LR_DECAY_ITERS', help='learning rate decay iterations')
parser.add_argument('--max-iters', default=600000, type=int,
                    metavar='MAX_ITERS', help='max iterations')
parser.add_argument('--eval-iters', default=100, type=int,
                    help='number of iterations to run eval on (default: 100)')
# dropout
parser.add_argument('--dropout', default=0.1, type=float, metavar='DROPOUT',
                    help='dropout')
# amp
parser.add_argument('--amp', default="bf16-mixed", type=str,
                    metavar='AMP', help='amp mode: [16-mixed|bf16-mixed]')
parser.add_argument('--compile', action='store_true', help='compile')
parser.add_argument('-p', '--log-interval', default=5, type=int,
                    metavar='N', help='log interval (default: 5)')
parser.add_argument('-g', '--gpu', default=None, type=str,
                    metavar='GPU', help='gpu (default None)')



# -----------------------------------------------------------------------------
args = parser.parse_args()
if args.config_file:
    # with open(args.config_file) as f:
    #     print(f.read())
    exec(open(args.config_file).read())
print(args)
# -----------------------------------------------------------------------------

best_acc1 = 0
torch.set_float32_matmul_precision('medium')

def configure_optimizers(model, weight_decay, learning_rate, betas):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    print(f"using fused AdamW: {use_fused}")
    return optimizer


def main():
    global best_acc1
    wandb_inited = False

    strategy = "auto"

    if args.gpu:
        accelerator = "cuda"
        devices = args.gpu
    else:
        accelerator = "auto"
        devices = "auto"

    precision = args.amp

    fabric = L.Fabric(precision=precision, strategy=strategy,
                      accelerator=accelerator, devices=devices)
    fabric.launch()

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model
    if args.arch == 'gpt2':
        config = GPTConfig()
        model = GPT2(config)
    elif args.arch == 'mamba':
        config = MyMambaConfig()
        model = MyMamba(config)
    else:
        raise ValueError(f"Invalid architecture: {args.arch}")
    model = model.cuda()

    # optimizer
    optimizer = configure_optimizers(model,
                                    args.weight_decay,
                                    args.learning_rate,
                                    (args.beta1, args.beta2))

    ############################################################
    # poor man's data loader
    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)
    # this came from 8,013,769 documents in total.
    train_data = np.memmap(os.path.join(args.data, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(args.data, 'val.bin'), dtype=np.uint16, mode='r')
    def get_batch(fabric, split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
        # move data to the same device as model
        x, y = x.pin_memory().to(fabric.device, non_blocking=True), y.pin_memory().to(fabric.device, non_blocking=True)
        return x, y

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(fabric):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                X, Y = get_batch(fabric, split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * it / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.lr_decay_iters:
            return args.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    # setup model and optimizer
    raw_model = model
    model, optimizer = fabric.setup(model, optimizer)

    # print(model)
    if args.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)
        print("compiled the model.")

    # train
    model.train()
    optimizer.zero_grad()
    assert args.gradient_accumulation_steps % fabric.world_size == 0
    gradient_accumulation_steps = args.gradient_accumulation_steps // fabric.world_size
    X, Y = get_batch(fabric, 'train') # fetch the very first batch

    start_time = time.time()
    iter_num = 0
    running_mfu = -0.001
    while True:

        iter_num += 1

        acc_loss = []
        for micro_step in range(gradient_accumulation_steps):

            # Accumulate gradient N batches at a time
            is_accumulating = micro_step != gradient_accumulation_steps - 1

            with fabric.no_backward_sync(model, enabled=is_accumulating):
                with fabric.autocast():
                    if args.arch == 'gpt2':
                        logits, loss = model(X, Y)
                    elif args.arch == 'mamba':
                        logits, loss = model(X, Y)
                        # outputs = model(input_ids=X, labels=Y)
                        # loss = outputs.loss
                        # logits = outputs.logits
                        # logits = model(X)
                        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)

                acc_loss.append(loss.item())

                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = get_batch(fabric, 'train')

                # loss = loss / gradient_accumulation_steps
                fabric.backward(loss / gradient_accumulation_steps)
                # fabric.backward(loss)

        # clip gradients
        if args.clip_gradients:
            fabric.clip_gradients(model, optimizer, clip_val=args.clip_gradients)

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Step the optimizer after accumulation phase is over
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        end_time = time.time()
        dt = end_time - start_time
        start_time = end_time

        # display
        if fabric.global_rank == 0 and (iter_num) % (args.log_interval) == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = numpy.mean(acc_loss) # * args.gradient_accumulation_steps
            mfu = raw_model.estimate_mfu(args.batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu < 0.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        if iter_num % (args.eval_interval) == 0 and fabric.global_rank == 0:
            losses = estimate_loss(fabric)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if args.wandb_log and fabric.global_rank == 0:
                # wandb init
                if (not wandb_inited) and args.wandb_log and fabric.global_rank == 0:
                    import wandb
                    merged_args = {**vars(args), **config.__dict__}
                    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
                    param_count = sum([np.prod(p.size()) for p in model_parameters])
                    args.wandb_run_name = f'{merged_args["arch"]}-{merged_args["n_layer"]}L-{param_count/1e6:.1f}M'
                    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=merged_args)
                    wandb_inited = True
                # wandb log
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })

        # termination conditions
        if iter_num > args.max_iters:
            break


if __name__ == '__main__':
    main()

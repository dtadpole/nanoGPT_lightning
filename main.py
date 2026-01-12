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
from ngpt import NGPT, NGPTConfig
from torch.profiler import profile, record_function, ProfilerActivity, ExecutionTraceObserver
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
# from fvcore.nn import FlopCountAnalysis
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
parser.add_argument('--strategy', default='auto', type=str,
                    help='strategy (default: auto)')
parser.add_argument('--profile', action='store_true', help='profile')
# learning rate
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
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
parser.add_argument('--eval-interval', default=200, type=int,
                    help='evaluation interval (default: 200)')
parser.add_argument('--wandb-log', action='store_true', help='enable wandb logging')
parser.add_argument('--wandb-project', default='gpt2', type=str,
                    help='wandb project name (default: gpt2)')
parser.add_argument('--wandb-run-name', default=None, type=str,
                    help='wandb run name (default: auto-generated)')



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

    if args.gpu:
        accelerator = "cuda"
        devices = args.gpu
    else:
        accelerator = "auto"
        devices = "auto"

    precision = args.amp

    fabric = L.Fabric(precision=precision, strategy=args.strategy,
                      accelerator=accelerator, devices=devices)
    fabric.launch()

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model
    if args.arch == 'gpt2':
        config = GPTConfig(
            block_size=args.block_size,
            dropout=args.dropout,
        )
        model = GPT2(config)
    elif args.arch == 'mamba':
        config = MyMambaConfig(
            block_size=args.block_size,
            dropout=args.dropout,
        )
        model = MyMamba(config)
    elif args.arch == 'ngpt':
        config = NGPTConfig(
            block_size=args.block_size,
            dropout=args.dropout,
        )
        model = NGPT(config)
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

    def profile_model(fabric, model):
        if not args.profile:
            return 0, 0
        print("Warming up...")
        # print(f"Fabric strategy: {fabric.strategy}")
        # warmup
        X, Y = get_batch(fabric, 'train')
        for _ in range(25):
            _, prof_loss = model(X, Y)
            X, Y = get_batch(fabric, 'train')
            fabric.backward(prof_loss)
        # profile
        print("DeepSpeed Profiling...")
        prof = FlopsProfiler(model)
        prof.start_profile()
        _, prof_loss = model(X, Y)
        X, Y = get_batch(fabric, 'train')
        fabric.backward(prof_loss)
        model_flops = prof.get_total_flops() / args.batch_size
        model_params = prof.get_total_params()
        print(f'Model flops: {model_flops/1e9:.1f}G')
        print(f'Model params: {model_params/1e6:.1f}M')
        # prof.print_model_profile(profile_step=5)
        prof.end_profile()
        print("DeepSpeed Profile ended.")

        # flops = FlopCountAnalysis(model, input)
        # model_flops_2 = flops.total()
        # print(f'Model flops: {model_flops_2/1e9:.1f}G')

        print("PyTorch Profiling...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                    # record_shapes=True, profile_memory=True, with_stack=True,
                    with_flops=True) as prof:
            for _ in range(2):
                prof.step() # Need to call this at each step to notify profiler of steps' boundary.
                _, prof_loss = model(X, Y)
                X, Y = get_batch(fabric, 'train')
                fabric.backward(prof_loss)
            print("PyTorch Profile done.")
        # print(prof.key_averages().table(sort_by="flops", row_limit=5))
        print("PyTorch Profile ended.")
        return model_flops, model_params

    def estimate_mfu(fwd_flops, fwdbwd_per_iter, dt):
        flops_per_fwdbwd = fwd_flops # * 3
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 82.5e12 # 4070 Ti Super GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

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

    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)
        print("compiled the model.")

    # profile the model
    # if fabric.global_rank == 0:
    model_flops, model_params = profile_model(fabric, model)
    if model_params == 0:
        model_params = sum(p.numel() for p in model.parameters())


    # train
    model.train()
    optimizer.zero_grad()
    assert args.gradient_accumulation_steps % fabric.world_size == 0
    gradient_accumulation_steps = args.gradient_accumulation_steps // fabric.world_size
    X, Y = get_batch(fabric, 'train') # fetch the very first batch


    start_time = time.time()
    iter_num = 0
    running_mfu_1 = -0.001
    running_mfu_2 = -0.001
    acc_loss = []
    # acc_loss2 = []
    while True:

        iter_num += 1

        for micro_step in range(gradient_accumulation_steps):

            # Accumulate gradient N batches at a time
            is_accumulating = micro_step != gradient_accumulation_steps - 1

            with fabric.no_backward_sync(model, enabled=is_accumulating):
                with fabric.autocast():
                    logits, loss = model(X, Y)

                acc_loss.append(loss.item())
                # acc_loss2.append(loss2.item())

                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = get_batch(fabric, 'train')

                # loss = loss / gradient_accumulation_steps
                # fabric.backward((loss + loss2) / gradient_accumulation_steps)
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

        # we have stepped optimizer, now call normalize_weights if nGPT
        if args.arch == 'ngpt':
            raw_model.normalize_weights()

        # measure elapsed time
        end_time = time.time()
        dt = end_time - start_time
        start_time = end_time

        # display
        if fabric.global_rank == 0 and (iter_num) % (args.log_interval) == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = numpy.mean(acc_loss) # * args.gradient_accumulation_steps
            acc_loss = [] # reset acc_loss
            # loss2f = numpy.mean(acc_loss2) # * args.gradient_accumulation_steps
            # acc_loss2 = [] # reset acc_loss2
            mfu_1 = estimate_mfu(model_flops, args.batch_size * gradient_accumulation_steps, dt)
            mfu_2 = model.estimate_mfu(fabric, args.batch_size * gradient_accumulation_steps, dt)
            running_mfu_1 = mfu_1 if running_mfu_1 < 0.0 else 0.9*running_mfu_1 + 0.1*mfu_1
            running_mfu_2 = mfu_2 if running_mfu_2 < 0.0 else 0.9*running_mfu_2 + 0.1*mfu_2
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu_1 {running_mfu_1*100:.2f}%, mfu_2 {running_mfu_2*100:.2f}%")

        if iter_num % (args.eval_interval) == 0 and fabric.global_rank == 0:
            losses = estimate_loss(fabric)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if args.wandb_log and fabric.global_rank == 0:
                # wandb init
                print("Wandb logging started...")
                if (not wandb_inited):
                    import wandb
                    merged_args = {**vars(args), **config.__dict__}
                    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
                    # param_count = sum([np.prod(p.size()) for p in model_parameters])
                    args.wandb_run_name = f'{merged_args["arch"]}-{merged_args["n_layer"]}L-{model_params/1e6:.1f}M'
                    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=merged_args)
                    wandb_inited = True
                # wandb log
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu_1*100, # convert to percentage
                })

        # termination conditions
        if iter_num > args.max_iters:
            break


if __name__ == '__main__':
    main()

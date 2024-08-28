import argparse
import os
import random
import shutil
import time
import math
import warnings
from enum import Enum

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

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DIR', nargs='?', default='./imagenet/',
                    help='path to dataset (default: ./imagenet/)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_b_16',
                    choices=model_names,
                    help='model architecture: (default: vit_b_16)')
parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                    help='number of data loading workers (default: 5)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup-epoch', default=5, type=float,
                    help='warmup epoch (default: 5)')
parser.add_argument('--restart-epoch', default=5, type=float,
                    help='restart epoch (default: 5)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-end', default=1e-5, type=float,
                    metavar='LREND', help='lr end')
parser.add_argument('--optimizer', default='AdamW', type=str, metavar='OPT',
                    help='optimizer [SGD|Adam|AdamW]')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--dropout', default=0.1, type=float, metavar='DROPOUT',
                    help='dropout')
# parser.add_argument('--stochastic_dropout', default=0.2, type=float,
#                     help='stochastic dropout (for MaxViT)')
parser.add_argument('--numops', default=2, type=int,
                    help='number of ops, default 2')
parser.add_argument('--magnitude', default=15, type=int,
                    help="magnitude (default: 15)")
parser.add_argument('--beta1', default=0.9, type=float, metavar='B1',
                    help='beta1')
parser.add_argument('--beta2', default=0.999, type=float, metavar='B2',
                    help='beta2')
parser.add_argument('--gamma', default=0.9, type=float, metavar='GAMMA',
                    help='gamma')
parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                    metavar='W', help='weight decay (default: 0.1)',
                    dest='weight_decay')
parser.add_argument('--scheduler', default='cosineR', type=str,
                    metavar='N', help='scheduler [step|exp|linear|cosine|cosineR]')
parser.add_argument('--amp', default="bf16-mixed", type=str,
                    metavar='AMP', help='amp mode: [16-mixed|bf16-mixed]')
parser.add_argument('--compile', action='store_true', help='compile')
parser.add_argument('--deepspeed', action='store_true', help='deepspeed')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('-g', '--gpu', default=None, type=str,
                    metavar='GPU', help='gpu (default None)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--dummy', action='store_true',
                    help="use fake data to benchmark")

best_acc1 = 0
torch.set_float32_matmul_precision('medium')


def main():
    global best_acc1

    args = parser.parse_args()
    print(args)

    if args.deepspeed:
        strategy = "deepspeed_stage_2"
    else:
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

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(
            1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(
            50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandAugment(num_ops=args.numops, magnitude=args.magnitude),
                # v2.MixUp(alpha=0.8, num_classes=1000),
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
            ]))

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('vit'):
            model = models.__dict__[args.arch](dropout=args.dropout, attention_dropout=args.dropout)
        # elif args.arch.startswith('maxvit'):
        #     model = models.__dict__[args.arch](stochastic_depth_prob=args.stochastic_dropout)
        else:
            model = models.__dict__[args.arch]()


    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     betas=(args.beta1, args.beta2))
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(args.beta1, args.beta2),
                                      weight_decay=args.weight_decay)
    else:
        raise Exception("unknown optimizer: ${args.optimizer}")

    # iters_per_epoch = math.ceil(len(train_loader) / args.batch_size)
    iters_per_epoch = math.ceil(
        len(train_dataset) / args.batch_size / fabric.world_size)
    print(f'iters_per_epoch: {iters_per_epoch}')

    if args.scheduler == 'step':
        """Sets the learning rate to the initial LR decayed by 0.9 every 2 epochs"""
        main_scheduler = StepLR(
            optimizer, step_size=2*iters_per_epoch, gamma=args.gamma)
    elif args.scheduler == 'exp':
        """Sets the learning rate to the initial LR decayed by 0.9 each epoch"""
        main_scheduler = ExponentialLR(
            optimizer, gamma=args.gamma)
    elif args.scheduler == 'linear':
        """Sets the learning rate to the initial LR and end LR"""
        main_scheduler = LinearLR(
            optimizer, start_factor=1.0, end_factor=args.lr_end/args.lr,
            total_iters=iters_per_epoch*args.epochs)
    elif args.scheduler == 'cosine':
        """Sets the learning rate to the initial LR and end LR"""
        main_scheduler = CosineAnnealingLR(
            optimizer, iters_per_epoch*(args.epochs-args.warmup_epoch), eta_min=args.lr_end)
    elif args.scheduler == 'cosineR':
        """Sets the learning rate to the initial LR and min LR and restart epochs"""
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer, iters_per_epoch*args.restart_epoch, eta_min=args.lr_end)
    else:
        raise Exception("unknown scheduler: ${args.scheduler}")

    # warm up with one epoch data
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-4,
                                end_factor=1.0, total_iters=math.ceil(iters_per_epoch*args.warmup_epoch))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler],
                             milestones=[math.ceil(iters_per_epoch*args.warmup_epoch)])

    # setup model and optimizer
    model, optimizer = fabric.setup(model, optimizer)

    # print(model)

    if args.compile:
        model = torch.compile(model)

    if fabric.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=False)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=True, drop_last=False)

        # data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler,
            prefetch_factor=5)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler,
            prefetch_factor=5)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
            prefetch_factor=5)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
            prefetch_factor=5)

    # evaluate only
    if args.evaluate:
        # evaluate
        validate(val_loader, model, criterion, fabric, args)
        exit(0)

    # fabric data loader
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    for epoch in range(args.start_epoch, args.epochs):

        batch_time = AverageMeter('Time', ':4.2f')
        data_time = AverageMeter('Data', ':4.2f')
        losses = AverageMeter('Loss', ':.2e')
        top1 = AverageMeter('Acc@1', ':5.2f')
        top5 = AverageMeter('Acc@5', ':5.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="[R:{}] [E:{}]".format(fabric.global_rank, epoch))

        # train
        model.train()

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            # move data to the same device as model
            images = images.to(fabric.device, non_blocking=True)
            target = target.to(fabric.device, non_blocking=True)

            learning_rate = f"LR {','.join(['%.2e' % e for e in scheduler.get_last_lr()])}"

            optimizer.zero_grad()
            output = model(images)
            with fabric.autocast():
                loss = criterion(output, target)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                fabric.backward(loss)
                optimizer.step()

            scheduler.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # display
            if (i+1) % args.print_freq == 0:
                progress.display(i + 1, optional=learning_rate)

        # display last batch
        progress.display(iters_per_epoch, optional=learning_rate)

        # evaluate
        validate(val_loader, model, criterion, fabric, args)


def validate(val_loader, model, criterion, fabric, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                # move data to the same device as model
                images = images.to(fabric.device, non_blocking=True)
                target = target.to(fabric.device, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if (i+1) % math.floor(args.print_freq / 2) == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    top1.all_reduce()
    top5.all_reduce()

    # if len(val_loader.sampler) * args.world_size < len(val_loader.dataset):
    #     aux_val_dataset = Subset(val_loader.dataset,
    #                              range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
    #     aux_val_loader = torch.utils.data.DataLoader(
    #         aux_val_dataset, batch_size=args.batch_size, shuffle=False,
    #         num_workers=args.workers, pin_memory=True)
    #     run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'{filename}.best')


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count],
                             dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, optional=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if optional is not None:
            entries.insert(3, optional)
        print('  '.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

import argparse
import os
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models import IR2Net_ResNet20

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def attention_erase(attention_maps, input_image, lamda=0.15):
    B,N,W,H = input_image.shape
    input = input_image
    batch_size, height, width = attention_maps.shape
    attention_maps = torch.nn.functional.interpolate(attention_maps.unsqueeze(1).detach(), size=(W, H), mode='bilinear', align_corners=True)
    weights = F.avg_pool2d(attention_maps, (W, H)).reshape(batch_size, -1)
    masks = []
    for i in range(batch_size):
        mask = attention_maps[i].detach()
        weight = weights[i]
        threshold = lamda * weight
        mask = (mask > threshold).float()
        masks.append(mask)
    masks = torch.stack(masks)
    erase_img = input*masks
    return erase_img

def at(x):
    return F.normalize(x.pow(2).mean(1))

def train(device, train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device, non_blocking=True)
        input_var = torch.autograd.Variable(input).to(device)
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # compute output
        output1, feature = model(input_var)
        feature = at(feature)
        erase_img = attention_erase(feature, input_var)
        output2, _ = model(erase_img)
        loss1 = criterion(output1, target_var)
        loss2 = criterion(output2, target_var)
        loss = (loss1 + loss2) / 2

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output1.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch+1, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    return top1.avg, losses.avg


def validate(device, val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).to(device)
            target_var = torch.autograd.Variable(target)

        if args.half:
            input_var = input_var.half()

        # compute output
        output, _ = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg

def main():
    torch.manual_seed(args.seed)

    use_gpu = not args.no_gpu and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_gpu else "cpu")
    gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = IR2Net_ResNet20(variation_range=args.variation_range).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if args.update_variation:
                model.update_variation(variation_range=args.variation_range)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']+1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              args.epochs,
                                                              eta_min=0,
                                                              last_epoch=-1)

    if args.evaluate:
        validate(device, val_loader, model, criterion)
        print('best_prec1 {:.3f}'.format(best_prec1))
        return

    best_prec1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_acc, train_loss = train(device, train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1, val_loss = validate(device, val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        log_message = str(epoch) + '\t' + '{:.3f}'.format(train_loss) + '\t' + '{:.3f}'.format(
            val_loss) + '\t' + '{:.3f}'.format(train_acc) + '\t' + '{:.3f}'.format(prec1) + '\t'
        logging.info(log_message)

    save_checkpoint({
        'epoch' : args.epochs,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, is_best, filename=os.path.join(args.save_dir, 'model.th'))
    print('best_prec1 {:.3f}'.format(best_prec1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.11, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=10)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disables GPU training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device to use (default: 0)')

    parser.add_argument('--variation-range', type=float, default=0.1,
                        help='the range of variation (default: 0.1)')
    parser.add_argument('--update-variation', action='store_true', default=False,
                        help='Update variation')

    args = parser.parse_args()

    main()

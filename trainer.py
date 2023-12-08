import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

from ray.tune.error import TuneError
from ray import tune

import nvsmi

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))
print_freq = 100


def train(args):
    cwd = os.path.dirname(os.path.realpath(__file__))
    datapath = os.path.join(cwd,'data')
    if args['arch'] not in model_names:
        raise TuneError("Unsupported model {}".format(args['arch']))

    model = torch.nn.DataParallel(resnet.__dict__[args['arch']]())
    model.cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=datapath, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args['batch_size'], shuffle=True,
        num_workers=args['workers'], pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=datapath, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args['workers'], pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args['half']:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args['lr'],
                                momentum=args['momentum'],
                                weight_decay=args['weight_decay'])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[args['epochs']//2,
                                                        2*args['epochs']//3],
                                                        gamma=0.1)
    if args['arch'] in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        warm_up_epochs=1
        lr_rollback=True

    for epoch in range(args['epochs']):

        # train for one epoch
        if args['arch'] in ['resnet1202', 'resnet110']:
            if warm_up_epochs>0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] =0.01
                warm_up_epochs-=1
            else:
                if lr_rollback:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args['lr']
                lr_rollback=False
        print('lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        epoch_train(train_loader, model, criterion, optimizer, epoch, args['half'])
        lr_scheduler.step()

        pid=os.getpid()
        gpu_process=None
        gpu_procs = nvsmi.get_gpu_processes()
        for proc in gpu_procs:
            if(proc.pid==pid):
                gpu_process=proc
                break
        if(gpu_process is not None):
            proc_name=gpu_process.process_name
            proc_pid=gpu_process.pid
            proc_usedmem=gpu_process.used_memory
        else:
            proc_name = 'N/A'
            proc_pid = 'N/A'
            proc_usedmem = 'N/A'

        # evaluate on validation set
        test_acc=validate(val_loader, model, criterion, args['half'])

        tune.report(
            Test_Accuracy=test_acc,
            Process_ID=proc_pid,
            Process_Name=proc_name,
            Process_Used_Memory=proc_usedmem
        )


def epoch_train(train_loader, model, criterion, optimizer, epoch, half):
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

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, half):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


if __name__ == '__main__':
    tune.run(train, resources_per_trial={"gpu":1},
             config={
              'arch':'resnet20',
              'workers':4,
              'epochs':1,
              'batch_size':tune.grid_search([256,512]),
              'lr':0.1,
              'momentum':0.9,
              'weight_decay':1e-4,
              'half':False})

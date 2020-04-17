import argparse
import os
import time
import logging
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models_bnn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from utils import *
from datetime import datetime

import dataset
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

# Logging
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder (named by datetime)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--seed', default=1234, type=int,
                    help='random seed')
# Model
parser.add_argument('--model', '-a', metavar='MODEL', default='mobilenetv1_1w1a',
                    help='model architecture ' )
parser.add_argument('--dataset',default='cifar10',type=str,
                    help='dataset')
parser.add_argument('--infl_ratio', default=1, type=float,
                    help='infl ratio of channels')
parser.add_argument('--num_classes', default=10, type=int,
                    help='number of classes')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
# Training
parser.add_argument('--gpus', default='1',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--lr', default=0.01,type=float,
                    help='learning rate')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--cache_size', default=50000, type=int,
                    help='cache size for data loader')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-bt','--batch_size_test', default=128, type=int,
                    help='mini-batch size for testing (default: 128)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--time-estimate', default=1, type=int,
                    metavar='N', help='print estimating finish time,set to 0 to disable')
parser.add_argument('--mixup', dest='mixup', action='store_true',
                    help='use mixup or not')
parser.add_argument('--labelsmooth', dest='labelsmooth', action='store_true',
                    help='use labelsmooth or not')


def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    random.seed(args.seed)
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = eval('models_bnn.'+args.model)().cuda()
    # model = torch.nn.DataParallel(eval('models_bnn.'+args.model)(),device_ids=[0, 1])
    model.L1_alpha = torch.nn.Parameter(torch.tensor(0.1), requires_grad=True)

    writer=SummaryWriter(os.path.join(save_path,'run'),comment='resnet'+args.model)
    logging.info("model structure: %s", model)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    if args.labelsmooth:
        criterion = LSR().cuda()
    else: 
        criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)().cuda()
    criterion.type(args.type)
    logging.info("criterion: %s", criterion)
    model.type(args.type)

    if args.half:
        model.half()
        criterion.half()

    train_loader, val_loader = dataset.load_data(
        dataset=args.dataset, batch_size=args.batch_size, batch_size_test=args.batch_size_test, num_workers=args.workers)
    # train_loader = get_cifar(type='train',batch_size=args.batch_size)
    # val_loader = get_cifar(type='val',batch_size=args.batch_size_test)


    if args.evaluate:
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0)
        logging.info('\n Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(val_loss=val_loss, val_prec1=val_prec1, val_prec5=val_prec5))
        return

    optimizer = torch.optim.SGD([{'params':model.parameters(),'initial_lr':args.lr}], args.lr,
                                momentum=0.9,
                                weight_decay=1e-5)
    # optimizer = torch.optim.Adam([{'params':model.parameters(),'initial_lr':args.lr}],lr=args.lr) 
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-5, eta_min = 1e-4, last_epoch=args.start_epoch)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min = 0, last_epoch=args.start_epoch)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [85,125], gamma=0.1, last_epoch=-1)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150,250,350], gamma=0.1, last_epoch=-1)
    logging.info('scheduler: %s', lr_scheduler)

    def cosin(i,T,emin=0,emax=0.01):
        return emin+(emax-emin)/2 * (1+np.cos(i*np.pi/T))

    def Log_UP(epoch):
        T_min, T_max = torch.tensor(1e-3).float(), torch.tensor(1e1).float()
        Tmin, Tmax = torch.log10(T_min).cuda(), torch.log10(T_max).cuda()
        threshold = 0
        if epoch<args.epochs-threshold:
            t = torch.tensor([torch.pow(torch.tensor(10.).cuda(), Tmin + (Tmax - Tmin) / (args.epochs-threshold) * epoch)]).float().cuda()
        else: 
            t = Tmax
        k = max(1/t,torch.tensor(1.).cuda()).float().cuda()
        return t, k

    #* Mixup
    beta_distribution = None
    if args.mixup:
        alpha=0.1
        beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
    #* setup conv_modules.epoch
    conv_modules=[]
    for name,module in model.named_modules():
        if isinstance(module,nn.Conv2d):
            conv_modules.append(module)
    for epoch in range(args.start_epoch, args.epochs):
        #optimizer = adjust_optimizer(optimizer, epoch, regime)
        time_start = datetime.now()
        if epoch <5: #*warm up
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * (epoch+1)/5
        for param_group in optimizer.param_groups:
            logging.info('lr: %s', param_group['lr'])

        # freeze k / t
        # t,k = Log_UP(epoch)
        # for name,module in model.named_modules():
        #     if isinstance(module,nn.Conv2d):
        #         module.k = k
        #         module.t = t
        for module in conv_modules:
            module.epoch=epoch
        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer,beta_distribution)

        #*****************LR_adjusting
        if epoch>=0:
            lr_scheduler.step()
        #****************

        # evaluate on validation set
        with torch.no_grad():
            for module in conv_modules:
                module.epoch=-1
            val_loss, val_prec1, val_prec5 = validate(
                val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = max(val_prec1, best_prec1)
            best_epoch = epoch
            best_loss = val_loss

        # save model checkpoint every few epochs
        if epoch % 1 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'parameters': list(model.parameters()),
            }, is_best, path=save_path)

        if args.time_estimate>0 and epoch%args.time_estimate==0:
           time_end = datetime.now()
           cost_time,finish_time = get_time(time_end-time_start,epoch,args.epochs)
           logging.info('Time cost: '+cost_time+'\t'
                        'Time of Finish: '+finish_time)

        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))
      
        writer.add_scalar('Train_loss',train_loss,epoch)
        writer.add_scalar('Train_prec_1',train_prec1,epoch)
        writer.add_scalar('Train_prec_5',train_prec5,epoch)
        writer.add_scalar('Val_loss',val_loss,epoch)
        writer.add_scalar('Val_prec_1',val_prec1,epoch)
        writer.add_scalar('Val_prec_5',val_prec5,epoch)
        for param_group in optimizer.param_groups:
            writer.add_scalar('LR',param_group['lr'],epoch)
        # writer.add_scalar('L1_alpha',model.L1_alpha,epoch)
        
        #* Tracking weights
        if epoch%2==0:
            Tracking(model,epoch)

    logging.info('*'*50+'DONE'+'*'*50)
    logging.info('\n Best_Epoch: {0}\t'
                     'Best_Prec1 {prec1:.4f} \t'
                     'Best_Loss {loss:.3f} \t'
                     .format(best_epoch+1, prec1=best_prec1, loss=best_loss))
    writer.close()

def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None,beta_distribution=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda()
        input_var = Variable(inputs.type(args.type)) ###not training)
        target_var = Variable(target)
        
        if args.half: #* fp16
            input_var = input_var.half()
        
        if beta_distribution: #*mixup Loss
            lambda_ = beta_distribution.sample([]).item()
            index = torch.randperm(input_var.size(0)).cuda()
            mixed_images = lambda_ * input_var + (1 - lambda_) * input_var[index, :]
            label_a, label_b = target_var, target_var[index]

            # Mixup loss.
            output = model(mixed_images)
            loss = (lambda_ * criterion(output, label_a)
                    + (1 - lambda_) * criterion(output, label_b))
        else: #* normal Loss
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

        #* L1-norm
        # L1_norm = 0
        # T = 1e-4
        # conv_parameters = []
        # for name,param in model.named_parameters():
        #     if 'conv' in name and 'layer' in name: # all conv layers except the first one
        #         conv_parameters.append(param)
        # for param in conv_parameters:
        #     L1_norm += torch.sum(torch.abs(torch.abs(param)-model.L1_alpha))
        # loss = loss + T * L1_norm

        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, 
                             top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer,beta_distribution):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)

def Tracking(model,epoch,path=None):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    conv = []
    for name,module in model.named_modules():
        if isinstance(module,nn.Conv2d):
            add=module.weight.cuda().flatten().detach()
            conv.append(add.cpu().numpy())
    i = 1
    fig=plt.figure(figsize=(16, 12))
    fig.tight_layout(pad=0.1, w_pad=3.0, h_pad=3.0)
    for w in conv:
        plt.subplot(5,4,i)
        plt.hist(w,bins=100)
        i+=1
    if not os.path.exists('./images'):
        os.mkdir('./images')
    path = './images/'+str(epoch)+'.png'
    plt.savefig(path, bbi=300,bbox_inches = 'tight')
    plt.close()

    Rconv = []
    for name,module in model.named_modules():
        if isinstance(module,nn.Conv2d):
            add=getattr(module,'Rweight',module.weight).cuda().flatten().detach()
            Rconv.append(add.cpu().numpy())
    i = 1
    fig=plt.figure(figsize=(16, 12))
    fig.tight_layout(pad=0.1, w_pad=3.0, h_pad=3.0)
    for w in Rconv:
        plt.subplot(5,4,i)
        plt.hist(w,bins=100)
        i+=1
    if not os.path.exists('./images2'):
        os.mkdir('./images2')
    path = './images2/'+str(epoch)+'.png'
    plt.savefig(path, bbi=300,bbox_inches = 'tight')
    plt.close()




if __name__ == '__main__':
    main()

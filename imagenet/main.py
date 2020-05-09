import argparse
import os
import time
import logging
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models_bnn
import models
import numpy as np
from torch.autograd import Variable
from utils.options import args
from utils.common import *
from modules import *
from datetime import datetime 
import dataset


def main():
    global args, best_prec1, best_prec5, conv_modules
    best_prec1 = 0
    best_prec5 = 0

    random.seed(args.seed)
    if args.evaluate:
        args.results_dir = '/tmp'
    # if args.save is '':
    #     args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path,'config.txt'), 'w') as args_file:
        args_file.write(str(datetime.now())+'\n\n')
        for args_n,args_v in args.__dict__.items():
            args_v = '' if not args_v and not isinstance(args_v,int) else args_v
            args_file.write(str(args_n)+':  '+str(args_v)+'\n')

    setup_logging(os.path.join(save_path, 'logger.log'))

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    if args.dataset=='tinyimagenet':
        num_classes=200
        model_zoo = 'models.'
    elif args.dataset=='imagenet':
        num_classes=1000
        model_zoo = 'models.'
    elif args.dataset=='cifar10': 
        num_classes=10
        model_zoo = 'models_bnn.'
    elif args.dataset=='cifar100': 
        num_classes=100
        model_zoo = 'models_bnn.'

    if len(args.gpus)==1:
        model = eval(model_zoo+args.model)(num_classes=num_classes).cuda()
    else: 
        model = nn.DataParallel(eval(model_zoo+args.model)(num_classes=num_classes))
    logging.info("model structure: %s", model)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            logging.error('invalid checkpoint: {}'.format(args.evaluate))
        else: 
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
            best_prec5 = checkpoint['best_prec5']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    #* label smooth
    if args.labelsmooth:
        criterion = LSR().cuda()
    else: 
        criterion = nn.CrossEntropyLoss().cuda()
    criterion.type(args.type)
    logging.info("criterion: %s", criterion)
    model.type(args.type)

    if args.evaluate:
        val_loader = dataset.get_imagenet(
                    type='val',
                    image_dir=args.data_path,
                    batch_size=args.batch_size_test,
                    num_threads=args.workers,
                    crop=224,
                    device_id='cuda:0',
                    num_gpus=1)
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0)
        logging.info('\n Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(val_loss=val_loss, val_prec1=val_prec1, val_prec5=val_prec5))
        return

    if args.dataset=='imagenet':
        train_loader = dataset.get_imagenet(type='train',
                                    image_dir=args.data_path,
                                    batch_size=args.batch_size,
                                    num_threads=args.workers,
                                    crop=224,
                                    device_id='cuda:0',
                                    num_gpus=1)
        val_loader = dataset.get_imagenet(type='val',
                                    image_dir=args.data_path,
                                    batch_size=args.batch_size_test,
                                    num_threads=args.workers,
                                    crop=224,
                                    device_id='cuda:0',
                                    num_gpus=1)
    else: 
        train_loader, val_loader = dataset.load_data(
                                    dataset=args.dataset, 
                                    data_path=args.data_path,
                                    batch_size=args.batch_size, 
                                    batch_size_test=args.batch_size_test, 
                                    num_workers=args.workers)

    optimizer = torch.optim.SGD([{'params':model.parameters(),'initial_lr':args.lr}], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam([{'params':model.parameters(),'initial_lr':args.lr}],lr=args.lr) 
    if args.lr_type == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = 0, last_epoch=args.start_epoch)
    elif args.lr_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=0.1, last_epoch=-1)
    logging.info('scheduler: %s', lr_scheduler)

    def cosin(i,T,emin=0,emax=0.01):
        "customized cos-lr"
        return emin+(emax-emin)/2 * (1+np.cos(i*np.pi/T))

    def Log_UP(epoch):
        "compute t&k in back-propagation"
        T_min, T_max = torch.tensor(args.Tmin).float(), torch.tensor(args.Tmax).float()
        Tmin, Tmax = torch.log10(T_min), torch.log10(T_max)
        t = torch.tensor([torch.pow(torch.tensor(10.), Tmin + (Tmax - Tmin) / args.epochs * epoch)]).float()
        k = max(1/t,torch.tensor(1.)).float()
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

    for epoch in range(args.start_epoch+1, args.epochs):
        time_start = datetime.now()
        #*warm up
        # if epoch <5: 
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = args.lr * (epoch+1)/5
        for param_group in optimizer.param_groups:
            logging.info('lr: %s', param_group['lr'])

        #* compute t/k in back-propagation
        t,k = Log_UP(epoch)
        for name,module in model.named_modules():
            if isinstance(module,nn.Conv2d):
                module.k = k.cuda()
                module.t = t.cuda()
        for module in conv_modules:
            module.epoch=epoch
        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer,beta_distribution)

        #* adjust Lr
        lr_scheduler.step()

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
            best_prec5 = max(val_prec5, best_prec5)
            best_epoch = epoch
            best_loss = val_loss

        # save model checkpoint every few epochs
        if epoch % 1 == 0:
            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            model_parameters = model.module.parameters() if len(args.gpus) > 1 else model.parameters()
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model_state_dict,
                'best_prec1': best_prec1,
                'best_prec5': best_prec5,
                'parameters': list(model_parameters),
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

        #* Tracking weights and plot histograms of weights 
        if args.weight_hist and epoch%args.weight_hist==0:
            Tracking(model,epoch,save_path)
        
        train_loader.reset()
        val_loader.reset()

    logging.info('*'*50+'DONE'+'*'*50)
    logging.info('\n Best_Epoch: {0}\t'
                     'Best_Prec1 {prec1:.4f} \t'
                     'Best_Prec5 {prec5:.4f} \t'
                     'Best_Loss {loss:.3f} \t'
                     .format(best_epoch+1, prec1=best_prec1, prec5=best_prec5, loss=best_loss))

def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None,beta_distribution=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, batch_data in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if i==1 and training:
            for module in conv_modules:
                module.epoch=-1
        inputs = batch_data[0]['data']
        target = batch_data[0]['label'].squeeze().long()
        batchsize = args.batch_size if training else args.batch_size_test
        len_dataloader = int(np.ceil(data_loader._size/batchsize))
        if args.gpus is not None:
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        input_var = Variable(inputs.type(args.type))
        target_var = Variable(target)
        
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
        # for no,(name,module) in model.named_modules():
        #     if no>1 and isinstance(module,nn.Conv2d): # all conv layers except the first one
        #         conv_parameters.append(module.Rweight)
        # for param in conv_parameters:
        #     L1_norm += torch.sum(torch.abs(torch.abs(param)-1))
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
                             epoch, i*batchsize, data_loader._size,
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

def Tracking(model,epoch,save_path=None):
    """
    plot histograms of weights
    """
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    conv = []
    for name,module in model.named_modules():
        if isinstance(module,nn.Conv2d):
            add=module.weight.cuda().flatten().detach()
            conv.append(add.cpu().numpy())
    img_size = int(np.ceil(len(conv)/5))
    i = 1
    fig=plt.figure(figsize=(16, 16*img_size/5))
    fig.tight_layout(pad=0.1, w_pad=3.0, h_pad=3.0)
    for w in conv:
        plt.subplot(img_size,5,i)
        plt.hist(w,bins=100)
        i+=1
    image_path=os.path.join(save_path, 'images')
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    path = os.path.join(image_path,str(epoch)+'.png')
    plt.savefig(path, bbi=300,bbox_inches = 'tight')
    plt.close()

    Rconv = []
    for name,module in model.named_modules():
        if isinstance(module,nn.Conv2d):
            add=getattr(module,'Rweight',module.weight).cuda().flatten().detach()
            Rconv.append(add.cpu().numpy())
    i = 1
    fig=plt.figure(figsize=(16, 16*img_size/5))
    fig.tight_layout(pad=0.1, w_pad=3.0, h_pad=3.0)
    for w in Rconv:
        plt.subplot(img_size,5,i)
        plt.hist(w,bins=100)
        i+=1
    image2_path=os.path.join(save_path, 'images2')
    if not os.path.exists(image2_path):
        os.mkdir(image2_path)
    path = os.path.join(image2_path,str(epoch)+'.png')
    plt.savefig(path, bbi=300,bbox_inches = 'tight')
    plt.close()


if __name__ == '__main__':
    main()

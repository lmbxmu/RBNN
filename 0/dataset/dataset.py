from datetime import datetime
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

def load_data(dataset='cifar10',batch_size = 256,batch_size_test=256,num_workers=0):
    # load data
    param = {'cifar10':{'name':datasets.CIFAR10,'size':32,'normalize':[[0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]]},
             'cifar100':{'name':datasets.CIFAR100,'size':96,'normalize':[(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)]},
             'mnist':{'name':datasets.MNIST,'size':96,'normalize':[(0.5,0.5,0.5),(0.5,0.5,0.5)]},
             'tinyimagenet':{'name':datasets.ImageFolder,'size':224,'normalize':[(0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)]}}
    data = param[dataset]

    if data['name']==datasets.ImageFolder:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(data['size']),
                # transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*data['normalize']),
            ]),
            'val': transforms.Compose([
                transforms.Resize(data['size']),
                transforms.ToTensor(),
                transforms.Normalize(*data['normalize']),
            ]),
            'test': transforms.Compose([
                transforms.Resize(data['size']),
                transforms.ToTensor(),
                transforms.Normalize(*data['normalize']),
            ])
        }
        data_dir = '/home/xuzihan/data/tiny-imagenet-200/'
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                        for x in ['train', 'val','test']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=100)
                        for x in ['train', 'val', 'test']}
        return dataloaders.values()

    else:
        transform1 = transforms.Compose([
            # transforms.Resize(data['size']),
            transforms.RandomCrop(data['size'],padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*data['normalize']),
        ])

        transform2 = transforms.Compose([
            #transforms.Resize(data['size']),
            transforms.ToTensor(),
            transforms.Normalize(*data['normalize']),
        ])

        trainset = data['name'](root='/home/xuzihan/data',
                                    train=True,
                                    download=False,
                                    transform=transform1);
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False)

        testset = data['name'](root='/home/xuzihan/data',
                                train=False,
                                download=False,
                                transform=transform2);
        testloader = DataLoader(
            testset,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=False)
        return trainloader, testloader

def delete_module_fromdict(statedict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k,v in statedict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

def add_module_fromdict(statedict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k,v in statedict.items():
        name = 'module.'+k
        new_state_dict[name] = v
    return new_state_dict

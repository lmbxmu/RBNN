from datetime import datetime
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def load_data(type='both',dataset='cifar10',data_path='/data',batch_size = 256,batch_size_test=256,num_workers=0):
    # load data
    param = {'cifar10':{'name':datasets.CIFAR10,'size':32,'normalize':[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]},
             'cifar100':{'name':datasets.CIFAR100,'size':32,'normalize':[(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)]},
             'mnist':{'name':datasets.MNIST,'size':32,'normalize':[(0.5,0.5,0.5),(0.5,0.5,0.5)]},
             'tinyimagenet':{'name':datasets.ImageFolder,'size':224,'normalize':[(0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)]}}
    data = param[dataset]

    if data['name']==datasets.ImageFolder:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(data['size']),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
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
        data_dir = os.path.join(data_path,'tiny-imagenet-200')
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                        for x in ['train', 'val']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=num_workers)
                        for x in ['train', 'val']}
        return dataloaders.values()

    else:
        transform1 = transforms.Compose([
            transforms.RandomCrop(data['size'],padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*data['normalize']),
        ])

        transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data['normalize']),
        ])

        trainset = data['name'](root=data_path,
                                    train=True,
                                    download=False,
                                    transform=transform1);
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)

        testset = data['name'](root=data_path,
                                train=False,
                                download=False,
                                transform=transform2);
        testloader = DataLoader(
            testset,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True)
        if type=='both':
            return trainloader, testloader
        elif type=='train':
            return trainloader
        elif type=='val':
            return testloader

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

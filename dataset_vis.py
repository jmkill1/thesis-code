'''Train CIFAR10 with PyTorch.'''
import torch

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


root_path = '../'
bs = 64
num_workers = 0
basesets = ['FashionMNIST', "MNIST", "CIFAR10"]
def _baseset_picker(baseset):
    size = 32
    if baseset == 'CIFAR10':
        ''' best transforms - figure out later (LF 06/11/21)
        '''
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
        clean_trainset = torchvision.datasets.CIFAR10(
            root=f'{root_path}/data', train=True, download=True, transform=transform_train)
        clean_trainloader = torch.utils.data.DataLoader(
            clean_trainset, batch_size=bs, shuffle=False, num_workers=num_workers)
        
        testset = torchvision.datasets.CIFAR10(
            root=f'{root_path}/data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=bs, shuffle=False, num_workers=num_workers)
    elif baseset == 'FashionMNIST':
        transform=transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        clean_trainset = torchvision.datasets.FashionMNIST(
            root=f'{root_path}/data', train=True, download=True, transform=transform)
        clean_trainloader = torch.utils.data.DataLoader(
            clean_trainset, batch_size=bs, shuffle=False, num_workers=num_workers)
        
        testset = torchvision.datasets.FashionMNIST(
            root=f'{root_path}/data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=bs, shuffle=False, num_workers=num_workers)
    elif baseset == 'MNIST':
        transform=transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        clean_trainset = torchvision.datasets.MNIST(
            root=f'{root_path}/data', train=True, download=True, transform=transform)
        clean_trainloader = torch.utils.data.DataLoader(
            clean_trainset, batch_size=bs, shuffle=False, num_workers=num_workers)
        
        testset = torchvision.datasets.MNIST(
            root=f'{root_path}/data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=bs, shuffle=False, num_workers=num_workers)

    return clean_trainloader, testloader, clean_trainset, testset

for baseset in basesets:
    clean_trainloader, testloader, clean_trainset, testset = _baseset_picker(baseset)
    images, label = next(iter(clean_trainloader))
    images_example = torchvision.utils.make_grid(images)
    images_example = images_example.numpy().transpose(1,2,0)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    images_example = images_example * std + mean
    plt.imshow(images_example)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'./{baseset}.png')



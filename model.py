import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import pickle

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from PIL import Image, ImageFilter

from .models import *
#from utils import progress_bar

print('==> Building model..')
def get_model(args, device):
    if args.net in ['ResNet','resnet']:
        net = ResNet18()
    elif args.net in ['VGG','vgg']:
        net = VGG('VGG19')
    elif args.net == 'GoogLeNet':
        net = GoogLeNet()
    elif args.net in ['DenseNet','densenet']:
        net = DenseNet121()
    elif args.net == 'MobileNet':
        net = MobileNetV2()
    elif args.net == 'LeNet':
        net = LeNet()
    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net = net.to(device)
    return net
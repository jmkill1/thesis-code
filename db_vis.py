'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import time

from model import get_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, get_random_images, get_noisy_images, AttackPGD
from evaluation import train, test, decision_boundary, test_on_adv
from options import options
from utils import simple_lapsed_time

args = options().parse_args()
print(args)
root_path = '../'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data/other training stuff
torch.manual_seed(args.set_data_seed)
trainloader, testloader = get_data(args)
torch.manual_seed(args.set_seed)

net = get_model(args, device)

# Load model
net.load_state_dict(torch.load(args.load_net, map_location=torch.device(device)))
    
start = time.time()
if args.imgs is None:
    #images, labels = get_random_images(trainloader.dataset)
    images, labels, image_ids = get_random_images(testloader.dataset)
else:
    # import ipdb; ipdb.set_trace()
    image_ids = args.imgs
    images = [trainloader.dataset[i][0] for i in image_ids]
    labels = [trainloader.dataset[i][1] for i in image_ids]
    print(labels)
# image_ids = args.imgs
sampleids = '_'.join(list(map(str,image_ids)))
# sampleids = '_'.join(list(map(str,labels)))
planeloader = make_planeloader(images, args)
preds = decision_boundary(args, net, planeloader, device)
from utils import produce_plot_alt,produce_plot_sepleg

net_name = args.net
plot_path = f'{root_path}/images/{net_name}/{str(args.baseset)}/{sampleids}'
os.makedirs(plot_path, exist_ok=True)
produce_plot_sepleg(args, plot_path, preds, planeloader, images, labels, trainloader, title = 'best', temp=1.0,true_labels = None)
produce_plot_alt(args, plot_path, preds, planeloader, images, labels, trainloader)

end = time.time()
simple_lapsed_time("Time taken to plot the image", end-start)
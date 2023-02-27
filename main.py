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
save_path = args.save_net
if args.active_log:
    import wandb
    idt = '_'.join(list(map(str,args.imgs)))
    wandb.init(project="Train_models", name = str(args.net) )
    wandb.config.update(args)

# Data/other training stuff
torch.manual_seed(args.set_data_seed)
trainloader, testloader = get_data(args)
torch.manual_seed(args.set_seed)
train_loss = []
test_accs = []
train_accs = []
net = get_model(args, device)

test_acc, predicted = test(args, net, testloader, device, 0)
print("scratch prediction ", test_acc)

criterion = get_loss_function(args)
if args.opt == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = get_scheduler(args, optimizer)

elif args.opt == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)



# Train or load base network
print("Training the network or loading the network")

start = time.time()
best_acc = 0  # best test accuracy
best_epoch = 0
if args.load_net is None:
    for epoch in range(args.epochs):
        train_acc, train_loss = train(args, net, trainloader, optimizer, criterion, device, args.train_mode, sam_radius=args.sam_radius)
        test_acc, predicted = test(args, net, testloader, device, epoch)
        print(f'EPOCH:{epoch}, Test acc: {test_acc}, Train_acc: {train_acc}, Train_loss: {train_loss}')
        if args.active_log:
            wandb.log({'epoch': epoch ,'test_accuracy': test_acc,
                       'train_acc': train_acc, 'train_loss': train_loss})
        if args.dryrun:
            break
        if args.opt == 'SGD':
            scheduler.step()

        # Save checkpoint.
        if test_acc > best_acc:
            print(f'The best epoch is: {epoch}')
            os.makedirs(f'{root_path}/ckp/{args.baseset}', exist_ok=True)
            
            print(f'{root_path}/ckp/{args.baseset}/{str(args.net)}.pth')
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(),
                            f'{root_path}/ckp/{args.baseset}/{str(args.net)}.pth')
            else:
                torch.save(net.state_dict(),
                            f'{root_path}/ckp/{args.baseset}/{str(args.net)}.pth')
            best_acc = test_acc
            best_epoch = epoch
else:
    net.load_state_dict(torch.load(args.load_net))
    

if args.load_net is None and args.active_log:
                wandb.log({'best_epoch': epoch ,'best_test_accuracy': best_acc
                    })
# test_acc, predicted = test(args, net, testloader, device)
# print(test_acc)

end = time.time()
simple_lapsed_time("Time taken to train/load the model", end-start)

if not args.plot_animation:
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
    produce_plot_sepleg(plot_path, preds, planeloader, images, labels, trainloader, args.baseset, title = 'best', temp=1.0,true_labels = None)
    produce_plot_alt(plot_path, preds, planeloader, images, labels, trainloader)

    end = time.time()
    simple_lapsed_time("Time taken to plot the image", end-start)
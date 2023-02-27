import torch
import torch.backends.cudnn as cudnn




from models import *
#from utils import progress_bar

print('==> Building model..')
def get_model(args, device):
    if args.net in ['ResNet','resnet']:
        net = ResNet18()
    elif args.net in ['VGG','vgg']:
        net = VGG('VGG19', args)
    elif args.net == 'GoogLeNet':
        net = GoogLeNet()
    elif args.net in ['DenseNet','densenet']:
        net = DenseNet121()
    elif args.net == 'MobileNet':
        net = MobileNetV2()
    elif args.net == 'LeNet':
        net = LeNet()
    elif args.net == 'AlexNet':
        net = AlexNet()
    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net = net.to(device)
    return net
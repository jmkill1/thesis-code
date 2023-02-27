# 在mnist数据集上面训练lenet
python main.py --net LeNet --baseset MNIST --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5 --set_seed 1
# 在fashion-mnist数据集上面训练AlexNet Vgg
python main.py --net AlexNet --baseset FashionMNIST --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5 --set_seed 1
python main.py --net VGG --baseset FashionMNIST --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5 --set_seed 1
# 在cigar-10数据集上面训练vgg resnet googlenet densenet mobilenetv2
python main.py --net VGG --baseset CIFAR10 --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5 --set_seed 1
python main.py --net resnet --baseset CIFAR10 --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5 --set_seed 1
python main.py --net GoogLeNet --baseset CIFAR10 --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5 --set_seed 1
python main.py --net densenet --baseset CIFAR10 --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5 --set_seed 1
python main.py --net MobileNet --baseset CIFAR10 --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5 --set_seed 1


#可视化MNIST决策边界
python db_vis.py --net LeNet --load_net ../ckp/MNIST/LeNet.pth --baseset MNIST --imgs 30,72,42 --resolution 500 --range_l 0.5 --range_r 0.5
#可视化FashionMNIST决策边界
python db_vis.py --net AlexNet --load_net ../ckp/FashionMNIST/AlexNet.pth --baseset FashionMNIST --imgs 30,72,42 --resolution 500 --range_l 0.5 --range_r 0.5
python db_vis.py --net VGG --load_net ../ckp/FashionMNIST/VGG.pth --baseset FashionMNIST --imgs 30,72,42 --resolution 500 --range_l 0.5 --range_r 0.5

#可视化CIFAR10决策边界

python db_vis.py --net VGG --load_net ../ckp/CIFAR10/VGG.pth --baseset CIFAR10 --imgs 30,72,42 --resolution 500 --range_l 0.5 --range_r 0.5
python db_vis.py --net ResNet --load_net ../ckp/CIFAR10/ResNet.pth --baseset CIFAR10 --imgs 30,72,42 --resolution 500 --range_l 0.5 --range_r 0.5
python db_vis.py --net GoogLeNet --load_net ../ckp/CIFAR10/GoogLeNet.pth --baseset CIFAR10 --imgs 30,72,42 --resolution 500 --range_l 0.5 --range_r 0.5
python db_vis.py --net DenseNet --load_net ../ckp/CIFAR10/DenseNet.pth --baseset CIFAR10 --imgs 30,72,42 --resolution 500 --range_l 0.5 --range_r 0.5
python db_vis.py --net MobileNet --load_net ../ckp/CIFAR10/MobileNet.pth --baseset CIFAR10 --imgs 30,72,42 --resolution 500 --range_l 0.5 --range_r 0.5
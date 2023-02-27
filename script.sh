# 在mnist数据集上面训练lenet
python main.py --net LeNet --baseset MNIST --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5
# 在fashion-mnist数据集上面训练AlexNet Vgg
python main.py --net Alexnet --baseset FashionMNIST --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5
python main.py --net vgg --baseset MNIST --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5
# 在cigar-10数据集上面训练vgg resnet googlenet densenet mobilenetv2
python main.py --net vgg --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5
python main.py --net resnet --baseset MNIST --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5
python main.py --net GoogLeNet --baseset MNIST --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5
python main.py --net densenet --baseset MNIST --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5
python main.py --net MobileNet --baseset MNIST --imgs 30,72,42 --resolution 500 --active_log --epochs 60 --lr 0.01 --range_l 0.5 --range_r 0.5
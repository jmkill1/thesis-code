import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchviz
from torch.autograd import Variable
import hiddenlayer as h
from torchsummary import summary

n_row = 1
img_size = 32
latent_dim = 100
n_classes = 10
channels = 1
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        # gen_input = torch.mul(self.label_emb(labels), noise)
        gen_input = torch.ones((1,100))
        print(gen_input.shape)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
 



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
# discriminator.eval()

# print(generator)
# print(discriminator)



z = Variable(torch.FloatTensor(np.random.normal(0, 1, (n_row, latent_dim))))
gen_labels = Variable(torch.LongTensor(np.random.randint(0, n_classes, n_row)))

summary(generator,(1,100))


print(z.shape)
print(gen_labels.shape)

# for param in generator.parameters():
#     print(param)


# gen_imgs = generator(z, gen_labels)
# print(gen_imgs.shape)
# validity, pred_label = discriminator(gen_imgs)



# g = torchviz.make_dot(gen_imgs, params=dict(list(generator.named_parameters())+[('z', z)]))
# g.view()
# g = torchviz.make_dot(pred_label, params=dict(list(discriminator.named_parameters())+[('gen_imgs', gen_imgs)]))
# g.view()


# vis_graph = h.build_graph(generator,args=(z,gen_labels))   # 获取绘制图像的对象
# vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
# vis_graph.save("./demo1.png")   # 保存图像的路径
# vis_graph = h.build_graph(discriminator,gen_imgs)   # 获取绘制图像的对象
# vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
# vis_graph.save("./demo2.png")   # 保存图像的路径
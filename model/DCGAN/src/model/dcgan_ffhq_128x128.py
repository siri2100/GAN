''' Reference
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dim_x, dim_z):
        super(Generator, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.deconv1 = nn.ConvTranspose2d(dim_z, 1024, 4, 1, 0, bias=False)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.deconv6 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self._weight_init()

    def forward(self, input):
        x = self.relu(self.bn1(self.deconv1(input)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.relu(self.bn5(self.deconv5(x)))
        x = self.tanh(self.deconv6(x))
        return x
    
    def _weight_init(self):
        nn.init.normal_(self.deconv1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.deconv2.weight.data, 0.0, 0.02)
        nn.init.normal_(self.deconv3.weight.data, 0.0, 0.02)
        nn.init.normal_(self.deconv4.weight.data, 0.0, 0.02)
        nn.init.normal_(self.deconv5.weight.data, 0.0, 0.02) 
        nn.init.normal_(self.deconv6.weight.data, 0.0, 0.02)
        nn.init.normal_(self.bn1.weight.data, 1.0, 0.02)
        nn.init.normal_(self.bn2.weight.data, 1.0, 0.02)
        nn.init.normal_(self.bn3.weight.data, 1.0, 0.02)
        nn.init.normal_(self.bn4.weight.data, 1.0, 0.02)
        nn.init.constant_(self.bn1.bias.data, 0.0)
        nn.init.constant_(self.bn2.bias.data, 0.0)
        nn.init.constant_(self.bn3.bias.data, 0.0)
        nn.init.constant_(self.bn4.bias.data, 0.0)
        nn.init.normal_(self.bn5.weight.data, 1.0, 0.02)
        nn.init.constant_(self.bn5.bias.data, 0.0)


class Discriminator(nn.Module):
    def __init__(self, dim_x):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(512, 1024, 4, 2, 1, bias=False)
        self.conv6 = nn.Conv2d(1024, 1, 4, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self._weight_init()
        
    def forward(self, input):
        x = self.lrelu(self.conv1(input))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.lrelu(self.bn5(self.conv5(x)))
        x = torch.sigmoid(self.conv6(x))
        return x

    def _weight_init(self):
        nn.init.normal_(self.conv1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.conv2.weight.data, 0.0, 0.02)
        nn.init.normal_(self.conv3.weight.data, 0.0, 0.02)
        nn.init.normal_(self.conv4.weight.data, 0.0, 0.02)
        nn.init.normal_(self.conv5.weight.data, 0.0, 0.02)
        nn.init.normal_(self.conv6.weight.data, 0.0, 0.02)
        nn.init.normal_(self.bn2.weight.data, 1.0, 0.02)
        nn.init.normal_(self.bn3.weight.data, 1.0, 0.02)
        nn.init.normal_(self.bn4.weight.data, 1.0, 0.02)
        nn.init.normal_(self.bn5.weight.data, 1.0, 0.02)
        nn.init.constant_(self.bn2.bias.data, 0.0)
        nn.init.constant_(self.bn3.bias.data, 0.0)
        nn.init.constant_(self.bn4.bias.data, 0.0)
        nn.init.constant_(self.bn5.bias.data, 0.0)

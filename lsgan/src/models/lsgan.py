import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dim_x, dim_z):
        super(Generator, self).__init__()
        self.dim_z = dim_z
        self.dim_x = dim_x
        self.fc = nn.Linear(dim_z, (8*dim_x)*(4**2))
        self.deconv1 = nn.ConvTranspose2d(8*dim_x, 4*dim_x, 4, 2, 1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(4*dim_x, 2*dim_x, 4, 2, 1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(2*dim_x, dim_x, 4, 2, 1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(dim_x, 3, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(4*dim_x)
        self.bn2 = nn.BatchNorm2d(2*dim_x)
        self.bn3 = nn.BatchNorm2d(dim_x)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self._weight_init()

    def forward(self, input):
        z = self.fc(input)
        z = z.view(z.shape[0], 512, self.dim_x//16, self.dim_x//16)
        x1 = self.relu(self.bn1(self.deconv1(z)))
        x2 = self.relu(self.bn2(self.deconv2(x1)))
        x3 = self.relu(self.bn3(self.deconv3(x2)))
        output = self.tanh(self.deconv4(x3))  
        return output
    
    def _weight_init(self):
        nn.init.normal_(self.deconv1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.deconv2.weight.data, 0.0, 0.02)
        nn.init.normal_(self.deconv3.weight.data, 0.0, 0.02)
        nn.init.normal_(self.deconv4.weight.data, 0.0, 0.02)
        nn.init.normal_(self.bn1.weight.data, 1.0, 0.02)
        nn.init.normal_(self.bn2.weight.data, 1.0, 0.02)
        nn.init.normal_(self.bn3.weight.data, 1.0, 0.02)
        nn.init.constant_(self.bn1.bias.data, 0.0)
        nn.init.constant_(self.bn2.bias.data, 0.0)
        nn.init.constant_(self.bn3.bias.data, 0.0)


class Discriminator(nn.Module):
    def __init__(self, dim_x):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, dim_x, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(dim_x, 2*dim_x, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(2*dim_x, 4*dim_x, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(4*dim_x, 8*dim_x, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(8*dim_x, 1, 4, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(2*dim_x)
        self.bn3 = nn.BatchNorm2d(4*dim_x)
        self.bn4 = nn.BatchNorm2d(8*dim_x)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self._weight_init()
        
    def forward(self, input):
        x1 = self.lrelu(self.conv1(input))
        x2 = self.lrelu(self.bn2(self.conv2(x1)))
        x3 = self.lrelu(self.bn3(self.conv3(x2)))
        x4 = self.lrelu(self.bn4(self.conv4(x3)))
        output = self.conv5(x4)
        return output

    def _weight_init(self):
        nn.init.normal_(self.conv1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.conv2.weight.data, 0.0, 0.02)
        nn.init.normal_(self.conv3.weight.data, 0.0, 0.02)
        nn.init.normal_(self.conv4.weight.data, 0.0, 0.02)
        nn.init.normal_(self.conv5.weight.data, 0.0, 0.02)  
        nn.init.normal_(self.bn2.weight.data, 1.0, 0.02)
        nn.init.normal_(self.bn3.weight.data, 1.0, 0.02)
        nn.init.normal_(self.bn4.weight.data, 1.0, 0.02)
        nn.init.constant_(self.bn2.bias.data, 0.0)
        nn.init.constant_(self.bn3.bias.data, 0.0)
        nn.init.constant_(self.bn4.bias.data, 0.0)

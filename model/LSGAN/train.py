import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model import lsgan_ffhq_128x128


''' Setting
        DATASET         MODEL               DIM_X
    01  FFHQ_128x128    dcgan_ffhq_128x128  128
'''
BATCH_SIZE = 128            # Batch Size
DATASET = 'FFHQ_128x128'    # CelebA, FFHQ_128x128, FFHQ_1024x1024
DEVICE = 'cuda'             # Computational Device : cpu, cuda
DIM_X = 64                  # Dimension of Image
DIM_Z = 1024                # Dimension of Latent Space 
DTYPE = 'torch.FloatTensor' # Floating Point Precision : torch.HalfTensor(fp16)(only for gpu), torch.FloatTensor(fp32)
EPOCH = 200                 # Target Train Epoch
EXP_NAME = 'exp1'           # Experiment Name
LR = 3e-4                   # Learning Rate
LOSS_A = -1
LOSS_B = 1
LOSS_C = 0
NUM_WORKER = 4              # Number of workers for dataloader


class Main:
    def __init__(self):
        # Step 00. Set Constant & Variable
        self.device = torch.device(DEVICE)

        # Step 01. Path
        self.path_parent = os.path.abspath('../..')
        self.path_model = f'{self.path_parent}/data/dst/LSGAN_{DATASET}_{EXP_NAME}/models'
        self.path_dataset = f'{self.path_parent}/data/src/{DATASET}'
        os.makedirs(f'{self.path_model}/tensorboard', exist_ok=True)
        
        # Step 02. Dataset
        train_set = datasets.ImageFolder(root=self.path_dataset,
                                         transform=transforms.Compose([
                                            transforms.Resize(DIM_X),
                                            transforms.CenterCrop(DIM_X),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ]))
        self.train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)

        # Step 03. Model
        if DATASET == 'FFHQ_128x128':
            self.generator = lsgan_ffhq_128x128.Generator(DIM_Z)
            self.discriminator = lsgan_ffhq_128x128.Discriminator()
        self.generator.to(self.device)
        self.generator.train()
        self.discriminator.to(self.device)
        self.discriminator.train()

        # Step 04. Loss
        self.loss = torch.nn.MSELoss()

        # Step 05. Optimizer
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=LR, betas=(0.5, 0.99), eps=1e-8)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=LR, betas=(0.5, 0.99), eps=1e-8)

        # Step 06. Tensorboard
        self.writer = SummaryWriter(f'{self.path_model}/tensorboard')

    def __call__(self, epoch):
        epoch_model = f'{epoch + 1}'.rjust(3, "0")

        for step, img in enumerate(tqdm((self.train_loader))):
            real_img = img[0].type(DTYPE)
            real_img = real_img.to(self.device)
            
            # Step 1. Update Discriminator
            self.discriminator.zero_grad()
            real_lbl = torch.full((real_img.size(0),), LOSS_B, dtype=torch.float, device=DEVICE)
            real_out = self.discriminator(real_img).view(-1)
            loss_d_real = 0.5*self.loss(real_out, real_lbl)
            loss_d_real.backward()
            fake_lbl = torch.full((real_img.size(0),), LOSS_A, dtype=torch.float, device=DEVICE) # DCGAN, EvolutionGAN_DCGAN
            latent_z = torch.randn(real_img.size(0), DIM_Z).type(DTYPE)
            latent_z = latent_z.to(self.device)
            fake_img = self.generator(latent_z)
            fake_out = self.discriminator(fake_img.detach()).view(-1)
            loss_d_fake = 0.5*self.loss(fake_out, fake_lbl)
            loss_d_fake.backward()
            loss_d = loss_d_real + loss_d_fake
            self.optimizer_d.step()
            
            # Step 2. Update Generator
            self.generator.zero_grad()
            fake_lbl = torch.full((real_img.size(0),), LOSS_C, dtype=torch.float, device=DEVICE)
            fake_out = self.discriminator(fake_img).view(-1)
            loss_g = 0.5*self.loss(fake_out, fake_lbl)
            loss_g.backward()
            self.optimizer_g.step()

            self.writer.add_scalar('discriminator_loss', loss_d, epoch*len(self.train_loader) + step)
            self.writer.add_scalar('generator_loss', loss_g, epoch*len(self.train_loader) + step)

        # Step 3. Save Generator
        torch.save(self.generator.state_dict(), f'{self.path_model}/generator_{epoch_model}.pth')


if __name__ == '__main__':
    model = Main()
    for idx_epoch in range(EPOCH):
        model(idx_epoch)

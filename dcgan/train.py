import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from src.model import dcgan

EXP_NAME = 'exp1'           # Experiment Name
EPOCH = 300                 # Target Train Epoch
BATCH_SIZE = 128            # Batch Size
DATASET = 'CelebA'          # CelebA, FFHQ_128x128, FFHQ_1024x1024
DEVICE = 'cuda'             # Computational Device : cpu, cuda
LR = 3e-4                   # Learning Rate
NUM_WORKER = 4              # Number of workers for dataloader

DIM_X = 64                  # Dimension of Image
DIM_Z = 100                 # Dimension of Latent Space 
DTYPE = 'torch.FloatTensor' # Floating Point Precision : torch.HalfTensor(fp16)(only for gpu), torch.FloatTensor(fp32)


class Main:
    def __init__(self):
        # Step 00. Set Constant & Variable
        self.device = torch.device(DEVICE)

        # Step 01. Path
        self.path_parent = os.path.abspath('../..')
        self.path_model = f'{self.path_parent}/data/dst/DCGAN_{EXP_NAME}/models'
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
        self.generator = dcgan.Generator(DIM_X, DIM_Z)
        self.generator.to(self.device)
        self.generator.train()
        self.discriminator = dcgan.Discriminator(DIM_X)
        self.discriminator.to(self.device)
        self.discriminator.train()

        # Step 04. Loss
        self.loss = torch.nn.BCELoss()

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
            real_lbl = torch.full((real_img.size(0),), 1, dtype=torch.float, device=DEVICE)
            real_out = self.discriminator(real_img).view(-1)
            loss_d_real = self.loss(real_out, real_lbl)
            loss_d_real.backward()
            fake_lbl = torch.full((real_img.size(0),), 0, dtype=torch.float, device=DEVICE)
            latent_z = torch.randn(real_img.size(0), DIM_Z, 1, 1).type(DTYPE)
            latent_z = latent_z.to(self.device)
            fake_img = self.generator(latent_z)
            fake_out = self.discriminator(fake_img.detach()).view(-1)
            loss_d_fake = self.loss(fake_out, fake_lbl)
            loss_d_fake.backward()
            loss_d = loss_d_real + loss_d_fake
            self.optimizer_d.step()
            
            # Step 2. Update Generator
            self.generator.zero_grad()
            fake_lbl = torch.full((real_img.size(0),), 1, dtype=torch.float, device=DEVICE)
            fake_out = self.discriminator(fake_img).view(-1)
            loss_g = self.loss(fake_out, fake_lbl)
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

import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.model import lsgan


EXP_NAME = 'exp1'
EPOCH_INIT = 1
EPOCH_FINE = 1
NUM_FAKE_IMG = 10
DATASET = 'CelebA'  # CelebA, FFHQ_128x128, FFHQ_1024x1024
DEVICE = 'cuda'     # Computational Device : cpu, cuda
NUM_WORKER = 2      # Number of workers for dataloader

# DCGAN
DIM_X = 64          # Dimension of Image
DIM_Z = 1024        # Dimension of Latent Space 
DTYPE = 'torch.FloatTensor'  # Floating Point Precision : torch.HalfTensor(fp16)(only for gpu), torch.FloatTensor(fp32)


class Main:
    def __init__(self, epoch):
        # Step 00. Set Constant & Variable
        self.epoch = f'{epoch}'.rjust(3, '0')
        self.device = torch.device(DEVICE)

        # Step 01. Path
        self.path_parent = os.path.abspath('../..')
        self.path_model = f'{self.path_parent}/data/dst/LSGAN_{EXP_NAME}/models/generator_{self.epoch}.pth'
        self.path_image_dst = f'{self.path_parent}/data/dst/LSGAN_{EXP_NAME}/images_{self.epoch}epoch'
        os.makedirs(self.path_image_dst, exist_ok=True)
      
        # Step 02. Model
        self.generator = lsgan.Generator(DIM_X, DIM_Z)
        self.generator.load_state_dict(torch.load(self.path_model, map_location=DEVICE))
        self.generator.eval()

    def __call__(self):
        for idx_img in tqdm(range(NUM_FAKE_IMG)):
            fake_img_num = f'{idx_img+1}'.rjust(3, '0')
            latent_z = torch.randn(1, DIM_Z).to(self.device)
            fake_img = self.generator(latent_z)
            fake_img = fake_img.squeeze().detach().cpu().numpy()
            fake_img = np.transpose(fake_img, (1, 2, 0))
            fake_img = fake_img * 255.0
            fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{self.path_image_dst}/{fake_img_num}.jpg', fake_img)


if __name__ == '__main__':
    for epoch in range(EPOCH_INIT, EPOCH_FINE + 1):
        model = Main(epoch)
        model()

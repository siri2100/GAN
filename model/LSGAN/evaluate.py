import os

import pandas as pd

from src.evaluation import fid


BATCH_SIZE = 8
DATASET = 'FFHQ_128x128'
DEVICE = 'cuda'
EXP_NAME = 'exp1'
NUM_WORKER = 0
EPOCH_INIT = 1
EPOCH_FINE = 1


csv_fid = []
csv_epoch = []

for i in range(EPOCH_INIT - 1, EPOCH_FINE):
    epoch = f'{i+1}'.rjust(3, '0')

    path_parent = os.path.abspath('../..')
    path_image_src = f'{path_parent}/data/src/{DATASET}/images'
    path_image_dst = f'{path_parent}/data/dst/LSGAN_{DATASET}_{EXP_NAME}/images_{epoch}epoch'
    fid_result = fid.main(path_image_src, path_image_dst, DEVICE, BATCH_SIZE, NUM_WORKER)
    csv_epoch.append(epoch)
    csv_fid.append(fid_result)
    print(f'epoch : {epoch} / fid : {fid_result}')

data = {
    'epoch': csv_epoch,
    'fid': csv_fid
}
df = pd.DataFrame(data)
df.to_csv(f'{path_parent}/data/dst/LSGAN_{DATASET}_{EXP_NAME}/performance_metric.csv')

import pdb
from tqdm import tqdm
import argparse
import yaml
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import trimesh
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop, SGD
from datasets.xhumans import XColor
from models.mlp import ColorAE


def main(config):
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    latent_feature = config['latent_feature']
    color_width = config['color_width']
    color_depth = config['color_depth']
    if_vae = config['if_vae']
    ae = ColorAE(latent_feature, color_width, color_depth, if_vae).cuda()
    dataset = XColor('/home/llx/xhumans', [18,20,25,27,34,35,41,85,87])
    loader = DataLoader(dataset, batch_size=1, shuffle=True,num_workers=8)

    print(len(dataset))
    optimizer = Adam(ae.parameters(), lr=0.001)

    pbar = tqdm(range(10))
    for i in pbar:
        for idx, data in tqdm(enumerate(loader)):
            color = data.cuda()

            # pdb.set_trace()
            if if_vae:
                pred_color, mu, log_var = ae(color)
                reg_loss = 0.01 * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2))
            else:
                pred_color, latent = ae(color)
                reg_loss = 10 * F.mse_loss(latent, torch.zeros_like(latent))

            color_loss = 10 * F.l1_loss(color, pred_color)
            loss = color_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            des = 'color:%.4f'%color_loss.item() + ' reg:%.4f'%reg_loss.item()
            pbar.set_description(des)

        with torch.no_grad():
            ret = np.random.choice(len(dataset))
            mesh = trimesh.load(dataset.obj_list[ret], process=False, maintain_order=True)

            img = np.array(mesh.visual.material.image)
            cv2.imwrite('gt%d.png'%i, img[:,:,::-1])
            color = torch.from_numpy((img/255.).astype(np.float32)).reshape(-1,3).cuda()

            if if_vae:
                pred_img,_,_ = ae(color)
            else:
                pred_img,_ = ae(color)
            pred_img = pred_img.detach().cpu().numpy().reshape(img.shape)
            pred_img = (pred_img*255).astype(np.uint8)
            cv2.imwrite('pred%d.png'%i, pred_img[:,:,::-1])

            print(peak_signal_noise_ratio(img, pred_img))
            torch.save(ae.state_dict(), 'data/color_ae.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/xhumans.yaml')
    args = parser.parse_args()
    main(args.config)
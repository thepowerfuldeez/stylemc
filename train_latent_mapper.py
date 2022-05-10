# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
import math
import time
import click
from typing import List, Optional

import wandb
import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD, Adam
from torchvision.transforms import Compose, Resize, CenterCrop
from PIL import Image

import legacy
import dnnlib
from id_loss import IDLoss
from landmarks_loss import LandmarksLoss, WingLoss
from mobilenet_facial import MobileNet_GDConv
from utils import read_image_mask, get_mean_std, generate_image, get_temp_shapes

from find_direction import init_clip_loss, compute_loss, denorm_img
from latent_mappers import Mapper

# 18 real, 8 for torgb layers
N_STYLE_CHANNELS = 26
S_NON_TRAINABLE_SPACE_CHANNELS = [0, 1, 4, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
S_TRAINABLE_SPACE_CHANNELS = [2, 3, 5, 6, 8, 9, 11, 12]


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=False,
              default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl")
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--s_input', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR',
              default='runs/male2female_id0.75_clip1.0_lr2.5_power2.0/')
@click.option('--text_prompt', help='Text for CLIP loss, also positive direction (if negative text prompt specified',
              type=str, required=True, default="a photo of a face of a feminine woman with no makeup")
@click.option('--negative_text_prompt', help='Negative direction of text prompt (use t1 - t2 for optimizing)',
              type=str, required=False, default="a photo of a face of a masculine man")
@click.option('--clip_type', help='Type of CLIP loss (small, large, double)', type=str, required=True, default='double')
@click.option('--clip_loss_type', help='Type of CLIP loss (nada or default)', type=str, required=True,
              default='default')
@click.option('--resolution', help='Resolution of output images', type=int, required=True, default=512)
@click.option('--batch_size', help='Batch Size', type=int, required=True, default=2)
@click.option('--learning_rate', help='Learning rate', type=float, required=True, default=0.0005)
@click.option('--n_epochs', help='number of epochs', type=int, required=True, default=10)
@click.option('--identity_loss_coef', help='Identity loss coef', type=float, required=True, default=0.3)
@click.option('--landmarks_loss_coef', help='Landmarks loss coef', type=float, required=True, default=0.0)
@click.option('--l2_reg_coef', help='l2 reg loss coef', type=float, required=True, default=0.8)
@click.option('--clip_loss_coef', help='CLIP loss coef', type=float, required=True, default=2.0)
def train_latent_mapper(
        ctx: click.Context,
        network_pkl: str,
        noise_mode: str,
        outdir: str,
        s_input: Optional[str],
        text_prompt: str,
        negative_text_prompt: Optional[str],
        clip_type: str,
        clip_loss_type: str,
        resolution: int,
        batch_size: int,
        learning_rate: float,
        n_epochs: int,
        identity_loss_coef: float,
        landmarks_loss_coef: float,
        l2_reg_coef: float,
        clip_loss_coef: float,
):
    wandb.init(project="stylegan2_latent_mapper", config=ctx.params)
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    os.makedirs(outdir, exist_ok=True)

    # for p in G.parameters():
    #     p.requires_grad = False
    mean, std = get_mean_std(device)
    img_size = 224
    transf = Compose([Resize(img_size, interpolation=Image.BICUBIC), CenterCrop(img_size)])

    styles_array = torch.tensor(np.load(s_input)['s'], device=device)
    n_items = styles_array.size(0)
    temp_shapes = get_temp_shapes(G)
    resolution_dict = {256: 6, 512: 7, 1024: 8}

    mapper = Mapper().to(device)

    checkpoint = torch.load("mobilenet_224_model_best_gdconv_external.pth.tar", map_location=device)
    mobilenet = torch.nn.DataParallel(MobileNet_GDConv(136)).to(device)
    mobilenet.load_state_dict(checkpoint['state_dict'])
    mobilenet.eval()

    landmarks_loss = LandmarksLoss()
    id_loss = IDLoss("a").to(device).eval()
    clip_loss1_func, clip_loss2_func = init_clip_loss(clip_loss_type, clip_type, device, text_prompt,
                                                      negative_text_prompt)

    temp_photos = []
    for i in range(math.ceil(n_items / batch_size)):
        # WARMING UP STEP
        # print(i*batch_size, "processed", time.time()-t1)

        styles_warmup = styles_array[i * batch_size:(i + 1) * batch_size].to(device)

        _, img2 = generate_image(G, resolution_dict[resolution], styles_warmup, temp_shapes, noise_mode)
        img2_cpu = img2.detach().cpu().numpy()
        temp_photos.append(img2_cpu)

    opt = Adam(mapper.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    num_batches = math.ceil(n_items / batch_size)
    total_num_iterations = num_batches * n_epochs
    cur_iteration = 0
    print(f"Total number of iterations: {total_num_iterations}")

    t1 = time.time()
    for epoch in range(n_epochs):
        for _ in range(num_batches):
            opt.zero_grad()
            cur_iteration += 1

            # # change learning rate param group of optimizer with cosine rule
            # new_learning_rate = np.cos(
            #     np.pi * cur_iteration / total_num_iterations) * learning_rate * 0.5 + learning_rate * 0.5
            # for param_group in opt.param_groups:
            #     param_group['lr'] = new_learning_rate

            i = np.random.randint(0, math.ceil(n_items / batch_size))
            styles = styles_array[i * batch_size:(i + 1) * batch_size].to(device)

            # new style vector
            styles_input = styles[:, S_TRAINABLE_SPACE_CHANNELS, :]  # batch x 8 x 512
            delta = mapper(styles_input)
            styles2 = styles.clone()
            styles2[:, S_TRAINABLE_SPACE_CHANNELS] += 0.1 * delta

            _, img = generate_image(G, resolution_dict[resolution], styles2, temp_shapes, noise_mode)

            # use original image for identity loss
            original_img = torch.tensor(temp_photos[i]).to(device)

            # ------ COMPUTE LOSS --------
            loss, loss_dict = compute_loss(
                img, original_img, transf, mean, std, device,
                clip_loss_type, clip_type, clip_loss_coef, clip_loss1_func, clip_loss2_func, text_prompt,
                negative_text_prompt,

                id_loss, identity_loss_coef,
                landmarks_loss, landmarks_loss_coef, mobilenet, img_size,
                styles, styles2, l2_reg_coef
            )
            # ------ COMPUTE LOSS --------

            if cur_iteration % 100 == 1:
                wandb.log({
                    "original_img": wandb.Image(np.stack([
                        denorm_img(original_img[i].detach().cpu()).numpy().astype('uint8')
                        for i in range(batch_size)
                    ])),
                    "generated_img": wandb.Image(np.stack([
                        denorm_img(img[i].detach().cpu()).numpy().astype('uint8')
                        for i in range(batch_size)
                    ])),
                    **loss_dict
                }, step=cur_iteration)

            loss.backward(retain_graph=True)

            grad_norm = 0
            for p in mapper.parameters():
                g = p.grad
                if g is not None:
                    grad_norm += g.data.norm()
            opt.step()

            if cur_iteration % 10 == 0:
                print(f"Iteration {cur_iteration}, img size: {img.size(-1)}, gradient norm: {grad_norm:.4f}")
                print(f"Total loss: {loss.item():.4f}, clip loss: {loss_dict['clip_loss']:.4f}, "
                      f"identity loss: {loss_dict['identity_loss']:.4f}, "
                      f"landmarks loss: {loss_dict['landmarks_loss']:.4f}, "
                      f"l2 loss: {loss_dict['l2_loss']:.4f}")

    output_mapper_filepath = f'{outdir}/mapper_{text_prompt.replace(" ", "_")}.pth'
    torch.save(mapper.state_dict(), output_mapper_filepath)

    print("time passed:", time.time() - t1)


if __name__ == "__main__":
    train_latent_mapper()

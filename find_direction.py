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

import clip
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop
from PIL import Image

import legacy
import dnnlib
from id_loss import IDLoss
from clip_loss import CLIPLoss
from utils import get_mean_std, generate_image, get_temp_shapes


# 18 real, 8 for torgb layers
N_STYLE_CHANNELS = 26
S_SPACE_CHANNELS = [0, 1, 4, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--s_input', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR',
              default='runs/male2female_id0.75_clip1.0_lr2.5_power2.0/')
@click.option('--text_prompt', help='Text for CLIP loss, also positive direction (if negative text prompt specified',
              type=str, required=True, default="a photo of a face of a feminine woman with no makeup")
@click.option('--negative_text_prompt', help='Negative direction of text prompt (use t1 - t2 for optimizing)',
              type=str, required=False, default="a photo of a face of a masculine man")
@click.option('--clip_type', help='Resolution of output images', type=str, required=True, default='small')
@click.option('--resolution', help='Resolution of output images', type=int, required=True, default=256)
@click.option('--batch_size', help='Batch Size', type=int, required=True, default=4)
@click.option('--learning_rate', help='Learning rate for s estimation, defaults to 2.5',
              type=float, required=True, default=2.5)
@click.option('--n_epochs', help='number of epochs', type=int, required=True, default=4)
@click.option('--identity_loss_coef', help='Identity loss coef', type=float, required=True, default=0.75)
@click.option('--clip_loss_coef', help='CLIP loss coef', type=float, required=True, default=1.0)
def find_direction(
        ctx: click.Context,
        network_pkl: str,
        noise_mode: str,
        outdir: str,
        s_input: Optional[str],
        text_prompt: str,
        negative_text_prompt: Optional[str],
        clip_type: str,
        resolution: int,
        batch_size: int,
        learning_rate: float,
        n_epochs: int,
        identity_loss_coef: float,
        clip_loss_coef: float,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    os.makedirs(outdir, exist_ok=True)

    # Generate images
    for i in G.parameters():
        i.requires_grad = True
    mean, std = get_mean_std(device)
    img_size = 224
    transf = Compose([Resize(img_size, interpolation=Image.BICUBIC), CenterCrop(img_size)])

    styles_input = np.load(s_input)['s']
    styles_array = torch.tensor(styles_input, device=device)
    n_items = styles_array.size(0)
    temp_shapes = get_temp_shapes(G)

    resolution_dict = {256: 6, 512: 7, 1024: 8}
    id_coeff = identity_loss_coef

    # trainable delta-s
    styles_direction = torch.zeros(1, N_STYLE_CHANNELS, 512, device=device)
    styles_direction.requires_grad_()

    id_loss = IDLoss("a").to(device).eval()
    clip_loss = CLIPLoss(text_prompt, device=device, negative_text_prompt=negative_text_prompt, clip_type=clip_type)

    temp_photos = []
    grads = []
    for i in range(math.ceil(n_items / batch_size)):
        # WARMING UP STEP
        # print(i*batch_size, "processed", time.time()-t1)

        styles = styles_array[i * batch_size:(i + 1) * batch_size].to(device)

        x2, img2 = generate_image(G, resolution_dict[resolution], styles, temp_shapes, noise_mode)
        img2_cpu = img2.detach().cpu().numpy()
        temp_photos.append(img2_cpu)

    t1 = time.time()
    for epoch in range(n_epochs):
        for i in range(math.ceil(n_items / batch_size)):
            styles = styles_array[i * batch_size:(i + 1) * batch_size].to(device)

            # new style vector
            styles2 = styles + styles_direction
            _, img = generate_image(G, resolution_dict[resolution], styles2, temp_shapes, noise_mode)

            # use original image for identity loss
            identity_loss, _ = id_loss(img, torch.tensor(temp_photos[i]).to(device))
            identity_loss *= id_coeff

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
            img = (transf(img.permute(0, 3, 1, 2)) / 255).sub_(mean).div_(std)
            clip_alignment_loss = clip_loss(img)
            clip_alignment_loss *= clip_loss_coef

            loss = identity_loss + clip_alignment_loss
            loss.backward(retain_graph=True)

            # set zeros to gradients on real s channels
            style_grad = styles_direction.grad
            style_grad[:, S_SPACE_CHANNELS, :] = 0
            # indices not in S_SPACE_CHANNELS
            grad_norm = style_grad[:, [j for j in range(N_STYLE_CHANNELS) if j not in S_SPACE_CHANNELS]].norm()

            styles_direction.data = (styles_direction - style_grad * learning_rate)
            grads.append(style_grad.clone())
            style_grad.data.zero_()

            print(f"Iteration {i}, img size: {img.size(-1)} , gradient norm: {grad_norm:.4f}")
            print(f"Clip loss: {clip_alignment_loss.item():.3f}, "
                  f"Identity loss: {identity_loss.item():.3f}, "
                  f"Total loss: {loss.item():.4f}")

    styles_direction = styles_direction.detach()
    output_direction_filepath = f'{outdir}/direction_{text_prompt.replace(" ", "_")}.npz'
    np.savez(output_direction_filepath, s=styles_direction.cpu().numpy())

    output_grads_filepath = f'{outdir}/grads_{text_prompt.replace(" ", "_")}.npz'
    np.savez(output_grads_filepath, grads=torch.stack(grads).cpu().numpy())

    print("time passed:", time.time() - t1)


if __name__ == "__main__":
    find_direction()

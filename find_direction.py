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
from utils import get_mean_std, generate_image, get_temp_shapes


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--s_input', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--text_prompt', help='Text', type=str, required=True)
@click.option('--resolution', help='Resolution of output images', type=int, required=True)
@click.option('--batch_size', help='Batch Size', type=int, required=True)
@click.option('--identity_power', help='How much change occurs on the face', type=str, required=True, default="low")
def find_direction(
        ctx: click.Context,
        network_pkl: str,
        noise_mode: str,
        outdir: str,
        s_input: Optional[str],
        text_prompt: str,
        resolution: int,
        batch_size: int,
        identity_power: str,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)
    label = torch.zeros([1, G.c_dim], device=device).requires_grad_()

    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize([text_prompt]).to(device)
    text_features = model.encode_text(text)

    # Generate images
    for i in G.parameters():
        i.requires_grad = True

    mean, std = get_mean_std(device)

    transf = Compose([Resize(224, interpolation=Image.BICUBIC), CenterCrop(224)])

    styles_input = np.load(s_input)['s']
    styles_array = torch.tensor(styles_input, device=device)
    n_items = styles_array.size(0)
    temp_shapes = get_temp_shapes(G)

    resolution_dict = {256: 6, 512: 7, 1024: 8}
    id_coeff_dict = {"high": 2, "medium": 0.5, "low": 0.15, "none": 0}
    id_coeff = id_coeff_dict[identity_power]
    styles_direction = torch.zeros(1, 26, 512, device=device)
    styles_direction_grad_el2 = torch.zeros(1, 26, 512, device=device)
    styles_direction.requires_grad_()

    global id_loss
    id_loss = IDLoss("a").to(device).eval()

    temp_photos = []
    grads = []
    for i in range(math.ceil(n_items / batch_size)):
        # WARMING UP STEP
        # print(i*batch_size, "processed", time.time()-t1)

        styles = styles_array[i * batch_size:(i + 1) * batch_size].to(device)

        x2, img2 = generate_image(G, resolution_dict[resolution], styles, temp_shapes, noise_mode)
        img2_cpu = img2.detach().cpu().numpy()
        temp_photos.append(img2_cpu)
        if i > 3:
            continue

        # new style vector
        styles2 = styles + styles_direction
        x, img = generate_image(G, resolution_dict[resolution], styles2, temp_shapes, noise_mode)

        identity_loss, _ = id_loss(img, img2)
        identity_loss *= id_coeff

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
        img = (transf(img.permute(0, 3, 1, 2)) / 255).sub_(mean).div_(std)
        image_features = model.encode_image(img)
        cos_sim = -1 * F.cosine_similarity(image_features, (text_features[0]).unsqueeze(0))

        (identity_loss + cos_sim.sum()).backward(retain_graph=True)

    t1 = time.time()
    # zero opt
    styles_direction.grad[:, list(range(26)), :] = 0
    with torch.no_grad():
        styles_direction *= 0

    for i in range(math.ceil(n_items / batch_size)):
        print(i * batch_size, "processed", time.time() - t1)

        styles = styles_array[i * batch_size:(i + 1) * batch_size].to(device)
        img2 = torch.tensor(temp_photos[i]).to(device)
        styles2 = styles + styles_direction

        x, img = generate_image(G, resolution_dict[resolution], styles2, temp_shapes, noise_mode)

        identity_loss, _ = id_loss(img, img2)
        identity_loss *= id_coeff

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
        img = (transf(img.permute(0, 3, 1, 2)) / 255).sub_(mean).div_(std)
        image_features = model.encode_image(img)
        cos_sim = -1 * F.cosine_similarity(image_features, (text_features[0]).unsqueeze(0))

        (identity_loss + cos_sim.sum()).backward(retain_graph=True)

        # why is that
        # styles_direction.grad[:, [0, 1, 4, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], :] = 0
        styles_direction.grad[:, [20, 21, 22, 23, 24, 25], :] = 0

        if i % 2 == 1:
            styles_direction.data = (styles_direction - styles_direction.grad * 5)
            grads.append(styles_direction.grad.clone())
            styles_direction.grad.data.zero_()
            if i > 3:
                styles_direction_grad_el2[grads[-2] * grads[-1] < 0] += 1

    styles_direction = styles_direction.detach()
    styles_direction[styles_direction_grad_el2 > (n_items / batch_size) / 4] = 0

    output_filepath = f'{outdir}/direction_' + text_prompt.replace(" ", "_") + '.npz'
    np.savez(output_filepath, s=styles_direction.cpu().numpy())

    print("time passed:", time.time() - t1)


if __name__ == "__main__":
    find_direction()

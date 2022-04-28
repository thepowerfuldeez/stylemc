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
from typing import List, Optional
from torchvision.transforms import Compose, Resize, CenterCrop
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import clip
from PIL import Image
import torch.nn.functional as F
import math
import time

import legacy
import dnnlib
from utils import get_mean_std, block_forward, num_range


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--projected_s', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--resolution', help='Resolution of output images', type=int, required=True)
@click.option('--batch_size', help='Batch Size', type=int, required=True)
@click.option('--identity_power', help='How much change occurs on the face', type=str, required=True)
def generate_images(
        ctx: click.Context,
        network_pkl: str,
        seeds: Optional[List[int]],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        class_idx: Optional[int],
        projected_w: Optional[str],
        projected_s: Optional[str],
        resolution: int,
        batch_size: int,
        identity_power: str
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device).requires_grad_()
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    model, preprocess = clip.load("ViT-B/32", device=device)

    text_prompts_file = open("text_prompts.txt")
    text_prompts = text_prompts_file.read().split("\n")
    text_prompts_file.close()

    text = clip.tokenize(text_prompts).to(device)
    text_features = model.encode_text(text)

    # Generate images
    for i in G.parameters():
        i.requires_grad = True

    mean, std = get_mean_std()

    transf = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
    ])

    styles_array = []
    print("seeds:", seeds)
    t1 = time.time()
    for seed_idx, seed in enumerate(seeds):
        if seed == seeds[-1]:
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        ws = G.mapping(z, label, truncation_psi=truncation_psi)

        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
            ws = ws.to(torch.float32)

            w_idx = 0
            for res in G.synthesis.block_resolutions:
                block = getattr(G.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        styles = torch.zeros(1, 26, 512, device=device)
        styles_idx = 0
        temp_shapes = []
        for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
            block = getattr(G.synthesis, f'b{res}')

            if res == 4:
                temp_shape = (block.conv1.affine.weight.shape[0], block.conv1.affine.weight.shape[0],
                              block.torgb.affine.weight.shape[0])
                styles[0, :1, :] = block.conv1.affine(cur_ws[0, :1, :])
                styles[0, 1:2, :] = block.torgb.affine(cur_ws[0, 1:2, :])
                if seed_idx == (len(seeds) - 1):
                    block.conv1.affine = torch.nn.Identity()
                    block.torgb.affine = torch.nn.Identity()
                styles_idx += 2
            else:
                temp_shape = (block.conv0.affine.weight.shape[0], block.conv1.affine.weight.shape[0],
                              block.torgb.affine.weight.shape[0])
                styles[0, styles_idx:styles_idx + 1, :temp_shape[0]] = block.conv0.affine(cur_ws[0, :1, :])
                styles[0, styles_idx + 1:styles_idx + 2, :temp_shape[1]] = block.conv1.affine(cur_ws[0, 1:2, :])
                styles[0, styles_idx + 2:styles_idx + 3, :temp_shape[2]] = block.torgb.affine(cur_ws[0, 2:3, :])
                if seed_idx == (len(seeds) - 1):
                    block.conv0.affine = torch.nn.Identity()
                    block.conv1.affine = torch.nn.Identity()
                    block.torgb.affine = torch.nn.Identity()
                styles_idx += 3
            temp_shapes.append(temp_shape)

        styles = styles.detach()
        styles_array.append(styles)

    resolution_dict = {256: 6, 512: 7, 1024: 8}
    identity_coefficient_dict = {"high": 2, "medium": 0.5, "low": 0.15, "none": 0}
    identity_coefficient = identity_coefficient_dict[identity_power]
    styles_wanted_direction = torch.zeros(1, 26, 512, device=device)
    styles_wanted_direction_grad_el2 = torch.zeros(1, 26, 512, device=device)
    styles_wanted_direction.requires_grad_()

    global id_loss
    id_loss = id_loss.IDLoss("a").to(device).eval()

    temp_photos = []
    grads = []
    for i in range(math.ceil(len(seeds) / batch_size)):
        # print(i*batch_size, "processed", time.time()-t1)

        styles = torch.vstack(styles_array[i * batch_size:(i + 1) * batch_size]).to(device)

        seed = seeds[i]

        styles_idx = 0
        x2 = img2 = None

        for k, (res, cur_ws) in enumerate(zip(G.synthesis.block_resolutions, block_ws)):
            block = getattr(G.synthesis, f'b{res}')
            if k > resolution_dict[resolution]:
                continue

            if res == 4:
                x2, img2 = block_forward(block, x2, img2, styles[:, styles_idx:styles_idx + 2, :], temp_shapes[k],
                                         noise_mode=noise_mode)
                styles_idx += 2
            else:
                x2, img2 = block_forward(block, x2, img2, styles[:, styles_idx:styles_idx + 3, :], temp_shapes[k],
                                         noise_mode=noise_mode)
                styles_idx += 3

        img2_cpu = img2.detach().cpu().numpy()
        temp_photos.append(img2_cpu)
        if i > 3:
            continue

        styles2 = styles + styles_wanted_direction

        styles_idx = 0
        x = img = None
        for k, (res, cur_ws) in enumerate(zip(G.synthesis.block_resolutions, block_ws)):
            block = getattr(G.synthesis, f'b{res}')
            if k > resolution_dict[resolution]:
                continue
            if res == 4:
                x, img = block_forward(block, x, img, styles2[:, styles_idx:styles_idx + 2, :], temp_shapes[k],
                                       noise_mode=noise_mode)
                styles_idx += 2
            else:
                x, img = block_forward(block, x, img, styles2[:, styles_idx:styles_idx + 3, :], temp_shapes[k],
                                       noise_mode=noise_mode)
                styles_idx += 3

        identity_loss, _ = id_loss(img, img2)
        identity_loss *= identity_coefficient
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
        img = (transf(img.permute(0, 3, 1, 2)) / 255).sub_(mean).div_(std)
        image_features = model.encode_image(img)
        cos_sim = -1 * F.cosine_similarity(image_features, (text_features[0]).unsqueeze(0))
        (identity_loss + cos_sim.sum()).backward(retain_graph=True)

    # t1 = time.time()

    for text_counter in range(len(text_prompts)):
        text_prompt = text_prompts[text_counter]
        print(text_prompt)

        styles_wanted_direction.grad.data.zero_()
        styles_wanted_direction_grad_el2 = torch.zeros(1, 26, 512, device=device)
        with torch.no_grad():
            styles_wanted_direction *= 0

        for i in range(math.ceil(len(seeds) / batch_size)):
            print(i * batch_size, "processed", time.time() - t1)

            styles = torch.vstack(styles_array[i * batch_size:(i + 1) * batch_size]).to(device)

            seed = seeds[i]

            img2 = torch.tensor(temp_photos[i]).to(device)

            styles2 = styles + styles_wanted_direction

            styles_idx = 0
            x = img = None
            for k, (res, cur_ws) in enumerate(zip(G.synthesis.block_resolutions, block_ws)):
                block = getattr(G.synthesis, f'b{res}')
                if k > resolution_dict[resolution]:
                    continue

                if res == 4:
                    x, img = block_forward(block, x, img, styles2[:, styles_idx:styles_idx + 2, :], temp_shapes[k],
                                           noise_mode=noise_mode)
                    styles_idx += 2
                else:
                    x, img = block_forward(block, x, img, styles2[:, styles_idx:styles_idx + 3, :], temp_shapes[k],
                                           noise_mode=noise_mode)
                    styles_idx += 3

            identity_loss, _ = id_loss(img, img2)
            identity_loss *= identity_coefficient
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
            img = (transf(img.permute(0, 3, 1, 2)) / 255).sub_(mean).div_(std)
            image_features = model.encode_image(img)
            cos_sim = -1 * F.cosine_similarity(image_features, (text_features[text_counter]).unsqueeze(0))
            (identity_loss + cos_sim.sum()).backward(retain_graph=True)

            styles_wanted_direction.grad[:, [0, 1, 4, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], :] = 0

            if i % 2 == 1:
                styles_wanted_direction.data = styles_wanted_direction - styles_wanted_direction.grad * 5
                grads.append(styles_wanted_direction.grad.clone())
                styles_wanted_direction.grad.data.zero_()

                if i > 3:
                    styles_wanted_direction_grad_el2[grads[-2] * grads[-1] < 0] += 1

        styles_wanted_direction_cpu = styles_wanted_direction.detach()
        styles_wanted_direction_cpu[styles_wanted_direction_grad_el2 > (len(seeds) / batch_size) / 4] = 0
        np.savez(f'{outdir}/direction_' + text_prompt.replace(" ", "_") + '.npz',
                 s=styles_wanted_direction_cpu.cpu().numpy())

    print("time passed:", time.time() - t1)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import time
import click
from typing import List, Optional

import torch
import numpy as np
from PIL import Image

import legacy
import dnnlib
from utils import get_temp_shapes, generate_image, block_forward, num_range


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--s_input', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--text_prompt', help='Text', type=str, required=True)
@click.option('--change_power', help='Change power', type=int, required=True)
def generate_images(
        ctx: click.Context,
        network_pkl: str,
        noise_mode: str,
        outdir: str,
        projected_w: Optional[str],
        s_input: Optional[str],
        text_prompt: str,
        change_power: int,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
            img = Image.fromarray(img[0].to(torch.uint8).cpu().numpy(), 'RGB')
            img.save(f'{outdir}/proj{idx:02d}.png')
        return


    # ----------------------------------------

    # Generate images
    for i in G.parameters():
        i.requires_grad = False

    t1 = time.time()
    temp_shapes = get_temp_shapes(G)

    if s_input is not None:
        styles = np.load(s_input)['s']
        styles = torch.tensor(styles, device=device)

        styles_direction = np.load(f'{outdir}/direction_' + text_prompt.replace(" ", "_") + '.npz')['s']
        styles_direction = torch.tensor(styles_direction, device=device)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    with torch.no_grad():
        for i in range(styles.shape[0]):
            imgs = []
            grad_changes = [0, change_power]

            for grad_change in grad_changes:
                styles += styles_direction * grad_change

                x, img = generate_image(G, 100, styles[[i]], temp_shapes, noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
                imgs.append(img[0].to(torch.uint8).cpu().numpy())

                styles -= styles_direction * grad_change

            img_filepath = f'{outdir}/{text_prompt.replace(" ", "_")}_{i:03d}.jpeg'
            Image.fromarray(np.concatenate(imgs, axis=1), 'RGB').save(img_filepath, quality=95)

        print("time passed:", time.time() - t1)


if __name__ == "__main__":
    generate_images()

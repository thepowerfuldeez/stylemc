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
from find_direction import S_TRAINABLE_SPACE_CHANNELS, N_STYLE_CHANNELS
from latent_mappers import Mapper
from run_deeplab import get_model, get_bg_mask, get_earring_mouth_lips_masks


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=False,
              default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl")
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--s_input', help='Projection result file', type=str, metavar='FILE')
@click.option('--use_mapper', help='use mapper or global direction', type=int, default=0)
@click.option('--n', help='generate first n results', type=int, metavar='FILE', default=99999)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--text_prompt', help='Text', type=str, required=True)
@click.option('--change_power', help='Change power', type=float, required=True, default=2.0)
@click.option('--use_blending', help='Perform segmentation + feature blending', type=int, required=True, default=0)
def generate_images(
        ctx: click.Context,
        network_pkl: str,
        noise_mode: str,
        outdir: str,
        projected_w: Optional[str],
        s_input: Optional[str],
        use_mapper: int,
        n: int,
        text_prompt: str,
        change_power: float,
        use_blending: int,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w'][:n]
        print(f"loaded {len(ws)} ws")
        ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
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
        styles = np.load(s_input)['s'][:n]
        styles = torch.tensor(styles, device=device)
        print(f"loaded {len(styles)} styles")

        if use_mapper:
            mapper_sd = torch.load(f'{outdir}/mapper_{text_prompt.replace(" ", "_")}.pth')
            mapper = Mapper()
            mapper.eval()
            mapper.load_state_dict(mapper_sd)
            mapper.to(device)
        else:
            global_styles_direction = np.load(f'{outdir}/direction_{text_prompt.replace(" ", "_")}.npz')['s']
            global_styles_direction = torch.tensor(global_styles_direction, device=device)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if use_blending:
        print("Using blending... Loading segmentation model")
        model_segm = get_model()
    else:
        model_segm = None

    with torch.no_grad():
        for i in range(styles.shape[0]):
            imgs = []
            grad_changes = [0, change_power]
            if i % 1000 == 0:
                print(i)

            masks_dict = {}
            xs_original = None

            for j, grad_change in enumerate(grad_changes):
                if use_mapper:
                    styles_direction = torch.zeros(1, N_STYLE_CHANNELS, 512, device=device)
                    with torch.no_grad():
                        delta = mapper(styles[i, S_TRAINABLE_SPACE_CHANNELS].unsqueeze(0))
                        styles_direction[:, S_TRAINABLE_SPACE_CHANNELS] = delta
                else:
                    styles_direction = global_styles_direction
                styles += styles_direction * grad_change

                xs, img = generate_image(G, 100, styles[[i]], temp_shapes, noise_mode, device,
                                         use_blending=use_blending, xs_original=xs_original, masks_dict=masks_dict)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
                img_arr = img[0].to(torch.uint8).cpu().numpy()

                # original image (grad_change == 0)
                if j == 0 and use_blending:
                    xs_original = xs
                    masks_dict['bg_mask'] = get_bg_mask(model_segm, img_arr, device)

                elif j == 1 and use_blending:
                    earring_mask, mouth_mask, teeth_mask = get_earring_mouth_lips_masks(
                        model_segm, img_arr, device,
                        need_earring_mask=not "face of a man" in text_prompt
                    )
                    masks_dict['earring_mask'] = earring_mask
                    masks_dict['mouth_mask'] = mouth_mask
                    masks_dict['teeth_mask'] = teeth_mask

                    assert 'bg_mask' in masks_dict
                    # after we have xs_original and computed masks for generated image, regenerate it again with blending
                    xs, img = generate_image(G, 100, styles[[i]], temp_shapes, noise_mode, device,
                                             use_blending=use_blending, xs_original=xs_original, masks_dict=masks_dict)
                    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
                    img_arr = img[0].to(torch.uint8).cpu().numpy()

                imgs.append(img_arr)

                styles -= styles_direction * grad_change

            img_filepath = f'{outdir}/{text_prompt.replace(" ", "_")}_{i:03d}.jpeg'
            Image.fromarray(np.concatenate(imgs, axis=1), 'RGB').save(img_filepath, quality=95)

        print("time passed:", time.time() - t1)


if __name__ == "__main__":
    generate_images()

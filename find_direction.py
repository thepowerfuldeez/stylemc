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
from torch.optim import AdamW, SGD, Adam
from torchvision.transforms import Compose, Resize, CenterCrop
from PIL import Image

import legacy
import dnnlib
from id_loss import IDLoss
from clip_loss import CLIPLoss
from landmarks_loss import LandmarksLoss
from utils import read_image_mask, get_mean_std, generate_image, get_temp_shapes
from mobilenet_facial import MobileNet_GDConv


# 18 real, 8 for torgb layers
N_STYLE_CHANNELS = 26
S_NON_TRAINABLE_SPACE_CHANNELS = [0, 1, 4, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
S_TRAINABLE_SPACE_CHANNELS = [2, 3, 5, 6, 8, 9, 11, 12]


def unprocess(img, transf, mean, std):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
    img = (transf(img.permute(0, 3, 1, 2)) / 255).sub_(mean).div_(std)
    return img


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
@click.option('--clip_type', help='Type of CLIP loss (small, large, double)', type=str, required=True, default='double')
@click.option('--only_face_mask', help='Perform optimization only at face region (excluded bg, hair, ears etc)',
              type=int, required=True, default=0)
@click.option('--mask_min_value', help='Mask min value when using face masks', type=float, required=True, default=0.1)
@click.option('--resolution', help='Resolution of output images', type=int, required=True, default=256)
@click.option('--batch_size', help='Batch Size', type=int, required=True, default=4)
@click.option('--learning_rate', help='Learning rate for s estimation, defaults to 2.5',
              type=float, required=True, default=1.5)
@click.option('--n_epochs', help='number of epochs', type=int, required=True, default=4)
@click.option('--identity_loss_coef', help='Identity loss coef', type=float, required=True, default=0.6)
@click.option('--landmarks_loss_coef', help='Landmarks loss coef', type=float, required=True, default=25.0)
@click.option('--l2_reg_coef', help='Landmarks loss coef', type=float, required=True, default=0.01)
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
        only_face_mask: int,
        mask_min_value: float,
        resolution: int,
        batch_size: int,
        learning_rate: float,
        n_epochs: int,
        identity_loss_coef: float,
        landmarks_loss_coef: float,
        l2_reg_coef: float,
        clip_loss_coef: float,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    os.makedirs(outdir, exist_ok=True)

    # Generate images
    for p in G.parameters():
        p.requires_grad = True
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

    trainable_delta_s = styles_direction.index_select(1, torch.tensor(S_TRAINABLE_SPACE_CHANNELS, device=device))
    trainable_delta_s.requires_grad = True

    checkpoint = torch.load("mobilenet_224_model_best_gdconv_external.pth.tar", map_location=device)
    mobilenet = torch.nn.DataParallel(MobileNet_GDConv(136)).to(device)
    mobilenet.eval()
    mobilenet.load_state_dict(checkpoint['state_dict'])
    landmarks_loss = LandmarksLoss()

    id_loss = IDLoss("a").to(device).eval()
    if clip_type == "double":
        clip_loss1 = CLIPLoss(text_prompt, device=device,
                              negative_text_prompt=negative_text_prompt, clip_type="small")
        clip_loss2 = CLIPLoss(text_prompt, device=device,
                              negative_text_prompt=negative_text_prompt, clip_type="large")
    else:
        clip_loss = CLIPLoss(text_prompt, device=device, negative_text_prompt=negative_text_prompt, clip_type=clip_type)

    if only_face_mask:
        idx = np.load(s_input)['idx']
        masks_dir = "/".join(s_input.split("/")[:-1]) + "/parsings/"
        masks_paths = [f"{masks_dir}/proj{idx[i]:02d}.png" for i in range(n_items)]
        masks = [read_image_mask(path, mask_min_value=mask_min_value, dilation=True) for path in masks_paths]

    temp_photos = []
    grads = []
    for i in range(math.ceil(n_items / batch_size)):
        # WARMING UP STEP
        # print(i*batch_size, "processed", time.time()-t1)

        styles = styles_array[i * batch_size:(i + 1) * batch_size].to(device)

        x2, img2 = generate_image(G, resolution_dict[resolution], styles, temp_shapes, noise_mode)
        img2_cpu = img2.detach().cpu().numpy()
        temp_photos.append(img2_cpu)

    opt = SGD([trainable_delta_s], lr=learning_rate)

    t1 = time.time()
    for epoch in range(n_epochs):
        for _ in range(math.ceil(n_items / batch_size)):
            opt.zero_grad()

            i = np.random.randint(0, math.ceil(n_items / batch_size))
            styles = styles_array[i * batch_size:(i + 1) * batch_size].to(device)

            # new style vector
            styles_direction[:, S_TRAINABLE_SPACE_CHANNELS] = trainable_delta_s
            styles2 = styles + styles_direction
            _, img = generate_image(G, resolution_dict[resolution], styles2, temp_shapes, noise_mode)

            # use original image for identity loss
            original_img = torch.tensor(temp_photos[i]).to(device)
            identity_loss, _ = id_loss(img, original_img)
            identity_loss *= id_coeff

            img = unprocess(img, transf, mean, std)

            with torch.no_grad():
                landmarks1 = mobilenet(unprocess(original_img, transf, mean, std))
                landmarks1 = landmarks1.view(landmarks1.size(0), -1, 2)
                landmarks2 = mobilenet(img)
                landmarks2 = landmarks2.view(landmarks2.size(0), -1, 2)
            face_landmarks_loss = landmarks_loss(landmarks1, landmarks2)
            face_landmarks_loss *= landmarks_loss_coef

            if only_face_mask:
                mask = torch.stack(masks[i * batch_size:(i + 1) * batch_size]).unsqueeze(1).to(device)
                mask = F.interpolate(mask, size=img.size()[2:], mode='bilinear', align_corners=False)
                img = img * mask

            if clip_type == "double":
                clip1_alignment_loss = clip_loss1(img)
                clip2_alignment_loss = clip_loss2(img)

                # when using double clip + only mask, no need to reduce second weight
                if only_face_mask:
                    clip_alignment_loss = clip1_alignment_loss + clip2_alignment_loss
                else:
                    clip_alignment_loss = clip1_alignment_loss + clip2_alignment_loss * 0.5
            else:
                clip_alignment_loss = clip_loss(img)
            clip_alignment_loss *= clip_loss_coef

            manipulation_direction_loss = trainable_delta_s.norm(2, dim=-1).mean()
            manipulation_direction_loss *= l2_reg_coef

            loss = identity_loss + clip_alignment_loss + manipulation_direction_loss + face_landmarks_loss
            loss.backward(retain_graph=True)

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_delta_s, max_norm=0.1)
            grads.append(trainable_delta_s.grad.clone())
            opt.step()

            print(f"Iteration {i}, img size: {img.size(-1)} , gradient norm: {grad_norm:.4f}")
            print(f"Clip loss: {clip_alignment_loss.item():.3f}, "
                  f"Identity loss: {identity_loss.item():.3f}, "
                  f"Landmarks loss: {face_landmarks_loss.item():.3f}, "
                  f"Manipulation direction loss: {manipulation_direction_loss.item():.3f}, "
                  f"Total loss: {loss.item():.4f}")

    styles_direction = styles_direction.detach()
    output_direction_filepath = f'{outdir}/direction_{text_prompt.replace(" ", "_")}.npz'
    np.savez(output_direction_filepath, s=styles_direction.cpu().numpy())

    output_grads_filepath = f'{outdir}/grads_{text_prompt.replace(" ", "_")}.npz'
    np.savez(output_grads_filepath, grads=torch.stack(grads).cpu().numpy())

    print("time passed:", time.time() - t1)


if __name__ == "__main__":
    find_direction()

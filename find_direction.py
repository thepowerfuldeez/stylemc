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
from clip_loss import CLIPLoss
from clip_loss_nada import CLIPLoss as CLIPLossNADA
from landmarks_loss import LandmarksLoss, WingLoss
from MTCNN import detect_faces
from mobilenet_facial import MobileNet_GDConv
from utils import read_image_mask, get_mean_std, generate_image, get_temp_shapes
from warp_images import crop_face

# 18 real, 8 for torgb layers
N_STYLE_CHANNELS = 26
S_NON_TRAINABLE_SPACE_CHANNELS = [0, 1, 4, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
S_TRAINABLE_SPACE_CHANNELS = [2, 3, 5, 6, 8, 9, 11, 12]


def denorm_img(img):
    img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255)
    return img


def unprocess(img, transf, mean, std):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
    img = (transf(img.permute(0, 3, 1, 2)) / 255).sub_(mean).div_(std)
    return img


def detect_landmarks(img, model, device, mean, std, out_size=224):
    """
    detect landmarks only for 1 image

    :param img: torch.Tensor after denorm_img function or List[torch.Tensor]
    :param model: mobilenet facial landmarks detector
    :return:
    """
    if isinstance(img, list):
        imgs = img
    else:
        imgs = [img]

    images_cropped = []
    orig_face_metas = []
    for img_ in imgs:

        # torch.Tensor -> numpy.ndarray
        faces, _ = detect_faces(img_, device=device)
        if len(faces):
            # torch.Tensor -> torch.Tensor
            cropped, orig_face_size, orig_bbox = crop_face(img_, faces, out_size)

            cropped_img = cropped.float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
            cropped_img = (cropped_img / 255 - mean.unsqueeze(0)) / std.unsqueeze(0)
            images_cropped.append(cropped_img)
            orig_face_metas.append((orig_face_size, orig_bbox))
        else:
            images_cropped.append(None)
            orig_face_metas.append(None)

    if any(x is None for x in images_cropped):
        return None

    img_batch = torch.cat(images_cropped, dim=0)
    with torch.no_grad():
        landmarks = model(img_batch)
    landmarks = landmarks.view(landmarks.size(0), -1, 2)

    for i, (orig_face_size, orig_bbox) in enumerate(orig_face_metas):
        landmarks[i] = landmarks[i] * orig_face_size + torch.tensor([orig_bbox[0], orig_bbox[1]],
                                                                    device=device).view(1, 2)
    return landmarks


def init_clip_loss(clip_loss_type, clip_type, device, text_prompt, negative_text_prompt):
    if clip_loss_type == "nada":
        if clip_type == "double":
            clip_loss1 = CLIPLossNADA(device, clip_model="ViT-B/32")
            clip_loss2 = CLIPLossNADA(device, clip_model="ViT-B/16")
        else:
            clip_loss1 = CLIPLossNADA(device, clip_model="ViT-B/32")
            clip_loss2 = None
    elif clip_loss_type == "nada_global":
        if clip_type == "double":
            clip_loss1 = CLIPLossNADA(device, clip_model="ViT-B/32", lambda_direction=0.0, lambda_global=1.0)
            clip_loss2 = CLIPLossNADA(device, clip_model="ViT-B/16", lambda_direction=0.0, lambda_global=1.0)
        else:
            clip_loss1 = CLIPLossNADA(device, clip_model="ViT-B/32", lambda_direction=0.0, lambda_global=1.0)
            clip_loss2 = None
    else:
        if clip_type == "double":
            clip_loss1 = CLIPLoss(device, text_prompt, negative_text_prompt, clip_type="small")
            clip_loss2 = CLIPLoss(device, text_prompt, negative_text_prompt, clip_type="large")
        else:
            clip_loss1 = CLIPLoss(device, text_prompt, negative_text_prompt, clip_type=clip_type)
            clip_loss2 = None
    return clip_loss1, clip_loss2


def compute_landmarks_loss(gen_img_batch, original_img_batch,
                           landmarks_loss, landmarks_loss_coef, model, device, mean, std, img_size=224):
    if landmarks_loss_coef != 0:
        try:
            landmarks1 = detect_landmarks([denorm_img(original_img_batch[i]) for i in range(len(original_img_batch))],
                                          model, device, mean, std, img_size)
        except:
            return 0

        try:
            landmarks2 = detect_landmarks([denorm_img(gen_img_batch[i]) for i in range(len(gen_img_batch))],
                                          model, device, mean, std, img_size)
            assert landmarks2 is not None
        except:
            print("could not detect landmarks")
            if landmarks1 is not None:
                landmarks2 = landmarks1
        face_landmarks_loss = landmarks_loss(landmarks1, landmarks2)
    else:
        face_landmarks_loss = 0
    return landmarks_loss_coef * face_landmarks_loss


def compute_clip_loss(gen_img_batch, original_img_batch, clip_loss_type, clip_type, clip_loss_coef,
                      clip_loss1, clip_loss2, transf, mean, std, device, text_prompt, negative_text_prompt):
    if clip_loss_type == "nada" or clip_loss_type == "nada_global":
        if clip_type == "double":
            clip1_alignment_loss = clip_loss1(original_img_batch, negative_text_prompt, gen_img_batch, text_prompt)
            clip2_alignment_loss = clip_loss2(original_img_batch, negative_text_prompt, gen_img_batch, text_prompt)

            clip_alignment_loss = clip1_alignment_loss + 0.5 * clip2_alignment_loss
        else:
            clip_alignment_loss = clip_loss1(original_img_batch, negative_text_prompt, gen_img_batch, text_prompt)
    else:
        img_unprocessed = unprocess(gen_img_batch, transf, mean, std)
        original_img_unprocessed = unprocess(original_img_batch, transf, mean, std)

        if clip_type == "double":
            clip1_alignment_loss = clip_loss1(original_img_unprocessed, img_unprocessed)
            clip2_alignment_loss = clip_loss2(original_img_unprocessed, img_unprocessed)

            clip_alignment_loss = clip1_alignment_loss + 0.5 * clip2_alignment_loss
        else:
            clip_alignment_loss = clip_loss1(original_img_unprocessed, img_unprocessed)
    return clip_loss_coef * clip_alignment_loss


def compute_loss(
        img, original_img, transf, mean, std, device,
        clip_loss_type, clip_type, clip_loss_coef, clip_loss1_func, clip_loss2_func, text_prompt, negative_text_prompt,
        id_loss, identity_loss_coef,
        landmarks_loss, landmarks_loss_coef, mobilenet,
        img_size, styles, styles2, l2_reg_coef
):
    identity_loss, _ = id_loss(img, original_img)
    identity_loss *= identity_loss_coef

    face_landmarks_loss = compute_landmarks_loss(img, original_img, landmarks_loss, landmarks_loss_coef,
                                                 mobilenet, device, mean, std, img_size)

    clip_alignment_loss = compute_clip_loss(
        img, original_img, clip_loss_type, clip_type, clip_loss_coef, clip_loss1_func, clip_loss2_func, transf,
        mean, std, device, text_prompt, negative_text_prompt
    )

    manipulation_direction_loss = l2_reg_coef * F.mse_loss(styles2[:, S_TRAINABLE_SPACE_CHANNELS],
                                                           styles[:, S_TRAINABLE_SPACE_CHANNELS])

    loss = identity_loss + clip_alignment_loss + manipulation_direction_loss + face_landmarks_loss
    loss_dict = {
        "clip_loss": clip_alignment_loss,
        "identity_loss": identity_loss,
        "landmarks_loss": face_landmarks_loss,
        "l2_loss": manipulation_direction_loss
    }
    return loss, loss_dict


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
@click.option('--resolution', help='Resolution of output images', type=int, required=True, default=256)
@click.option('--batch_size', help='Batch Size', type=int, required=True, default=4)
@click.option('--learning_rate', help='Learning rate for s estimation, defaults to 2.5',
              type=float, required=True, default=1.5)
@click.option('--n_epochs', help='number of epochs', type=int, required=True, default=4)
@click.option('--resume', help='resume checkpoint', type=str, required=False, default=None)
@click.option('--identity_loss_coef', help='Identity loss coef', type=float, required=True, default=0.6)
@click.option('--landmarks_loss_coef', help='Landmarks loss coef', type=float, required=True, default=25.0)
@click.option('--l2_reg_coef', help='Landmarks loss coef', type=float, required=True, default=0.1)
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
        clip_loss_type: str,
        resolution: int,
        batch_size: int,
        learning_rate: float,
        n_epochs: int,
        resume: str,
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

    mean, std = get_mean_std(device)
    img_size = 224
    transf = Compose([Resize(img_size, interpolation=Image.BICUBIC), CenterCrop(img_size)])

    styles_array = torch.tensor(np.load(s_input)['s'], device=device)
    n_items = styles_array.size(0)
    temp_shapes = get_temp_shapes(G)
    resolution_dict = {256: 6, 512: 7, 1024: 8}

    # trainable delta-s
    if resume:
        styles_direction = torch.tensor(np.load(resume)['s'], map_location=device)
        print(f"Loaded direction from {resume}")
    else:
        styles_direction = torch.zeros(1, N_STYLE_CHANNELS, 512, device=device)
    trainable_delta_s = styles_direction.index_select(1, torch.tensor(S_TRAINABLE_SPACE_CHANNELS, device=device))
    print(f"training param shape {trainable_delta_s.shape}")
    trainable_delta_s.requires_grad = True

    checkpoint = torch.load("mobilenet_224_model_best_gdconv_external.pth.tar", map_location=device)
    mobilenet = torch.nn.DataParallel(MobileNet_GDConv(136)).to(device)
    mobilenet.load_state_dict(checkpoint['state_dict'])
    mobilenet.eval()

    landmarks_loss = LandmarksLoss()  # WingLoss(omega=8)
    id_loss = IDLoss("a").to(device).eval()
    clip_loss1_func, clip_loss2_func = init_clip_loss(clip_loss_type, clip_type, device, text_prompt,
                                                      negative_text_prompt)

    opt = SGD([trainable_delta_s], lr=learning_rate)
    num_batches = math.ceil(n_items / batch_size)
    total_num_iterations = num_batches * n_epochs
    cur_iteration = 0
    print(f"Total number of iterations: {total_num_iterations}")

    t1 = time.time()
    for epoch in range(n_epochs):
        for _ in range(num_batches):
            opt.zero_grad()
            cur_iteration += 1

            # change learning rate param group of optimizer with cosine rule
            new_learning_rate = np.cos(
                np.pi * cur_iteration / total_num_iterations) * learning_rate * 0.5 + learning_rate * 0.5
            for param_group in opt.param_groups:
                param_group['lr'] = new_learning_rate

            i = np.random.randint(0, math.ceil(n_items / batch_size))
            styles = styles_array[i * batch_size:(i + 1) * batch_size].to(device)

            # new style vector
            styles_direction[:, S_TRAINABLE_SPACE_CHANNELS] = trainable_delta_s
            styles2 = styles + styles_direction
            _, img = generate_image(G, resolution_dict[resolution], styles2, temp_shapes, noise_mode)

            # use original image for identity loss
            _, original_img = generate_image(G, resolution_dict[resolution], styles, temp_shapes, noise_mode)

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
                    "original_img": wandb.Image(denorm_img(original_img[0].detach().cpu()).numpy().astype('uint8')),
                    "generated_img": wandb.Image(denorm_img(img[0].detach().cpu()).numpy().astype('uint8')),
                    **loss_dict
                }, step=cur_iteration)

            if cur_iteration % 1000 == 999:
                np.savez(f"{outdir}/direction_last.npz", s=styles_direction.detach().cpu().numpy())

            loss.backward(retain_graph=True)

            grad_norm = trainable_delta_s.grad.data.norm()
            opt.step()

            if cur_iteration % 10 == 0:
                print(f"Iteration {cur_iteration}, img size: {img.size(-1)}, gradient norm: {grad_norm:.4f}, "
                      f"lr {new_learning_rate:.4f}")
                print(f"Total loss: {loss.item():.4f}, clip loss: {loss_dict['clip_loss']:.4f}, "
                      f"identity loss: {loss_dict['identity_loss']:.4f}, "
                      f"landmarks loss: {loss_dict['landmarks_loss']:.4f}, "
                      f"l2 loss: {loss_dict['l2_loss']:.4f}")

    styles_direction = styles_direction.detach()
    output_direction_filepath = f'{outdir}/direction_{text_prompt.replace(" ", "_")}.npz'
    np.savez(output_direction_filepath, s=styles_direction.cpu().numpy())

    print("time passed:", time.time() - t1)


if __name__ == "__main__":
    find_direction()

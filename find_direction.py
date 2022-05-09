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


def detect_landmarks(img, model, device, mean, std, out_size):
    """
    detect landmarks only for 1 image

    :param img: torch.Tensor after denorm_img function
    :param model: mobilenet facial landmarks detector
    :return:
    """
    # torch.Tensor -> numpy.ndarray
    faces, _ = detect_faces(img)
    if len(faces):
        # torch.Tensor -> torch.Tensor
        cropped, orig_face_size, orig_bbox = crop_face(img, faces, out_size)

        img = cropped.float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
        img_batch = (img / 255 - mean.unsqueeze(0)) / std.unsqueeze(0)
        with torch.no_grad():
            landmarks = model(img_batch)
        landmarks = landmarks.view(landmarks.size(0), -1, 2)
        landmarks = landmarks * orig_face_size + torch.tensor([orig_bbox[0], orig_bbox[1]], device=device).view(1, 1, 2)
        return landmarks[0]
    else:
        return torch.zeros(68, 2, device=device)


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
        landmarks1 = torch.stack([
            detect_landmarks(denorm_img(original_img_batch[i]), model, device, mean, std, img_size)
            for i in range(original_img_batch.shape[0])
        ])
        try:
            landmarks2 = torch.stack([
                detect_landmarks(denorm_img(gen_img_batch[i]), model, device, mean, std, img_size)
                for i in range(gen_img_batch.shape[0])
            ])
        except:
            print("could not detect landmarks")
            landmarks2 = landmarks1
        face_landmarks_loss = landmarks_loss(landmarks1, landmarks2)
    else:
        face_landmarks_loss = 0
    return face_landmarks_loss * landmarks_loss_coef


def compute_clip_loss(gen_img_batch, original_img_batch, clip_loss_type, clip_type, clip_loss_coef,
                      clip_loss1, clip_loss2, transf, mean, std, device, text_prompt, negative_text_prompt):
    # if only_face_mask:
    #     mask = torch.stack(masks[i * batch_size:(i + 1) * batch_size]).unsqueeze(1).to(device)
    #     mask = F.interpolate(mask, size=img_unprocessed.size()[2:], mode='bilinear', align_corners=False)
    #     img_unprocessed = img_unprocessed * mask
    #     mask2 = mask.detach()
    #     mask2[:, int(mask.size(1) * 0.85):] = mask_min_value
    #     original_img_unprocessed = original_img_unprocessed * mask2

    if clip_loss_type == "nada" or clip_loss_type == "nada_global":
        if clip_type == "double":
            clip1_alignment_loss = clip_loss1(original_img_batch, negative_text_prompt, gen_img_batch, text_prompt)
            clip2_alignment_loss = clip_loss2(original_img_batch, negative_text_prompt, gen_img_batch, text_prompt)

            # when using double clip + only mask, no need to reduce second weight
            if 0: #only_face_mask:
                clip_alignment_loss = clip1_alignment_loss + clip2_alignment_loss
            else:
                clip_alignment_loss = clip1_alignment_loss + clip2_alignment_loss * 0.5
        else:
            clip_alignment_loss = clip_loss1(original_img_batch, negative_text_prompt, gen_img_batch, text_prompt)
    else:
        img_unprocessed = unprocess(gen_img_batch, transf, mean, std)
        original_img_unprocessed = unprocess(original_img_batch, transf, mean, std)

        if clip_type == "double":
            clip1_alignment_loss = clip_loss1(original_img_unprocessed, img_unprocessed)
            clip2_alignment_loss = clip_loss2(original_img_unprocessed, img_unprocessed)

            # when using double clip + only mask, no need to reduce second weight
            if 0: #only_face_mask:
                clip_alignment_loss = clip1_alignment_loss + clip2_alignment_loss
            else:
                clip_alignment_loss = clip1_alignment_loss + clip2_alignment_loss * 0.5
        else:
            clip_alignment_loss = clip_loss1(original_img_unprocessed, img_unprocessed)
    return clip_alignment_loss * clip_loss_coef


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
@click.option('--only_face_mask', help='Perform optimization only at face region (excluded bg, hair, ears etc)',
              type=int, required=True, default=0)
@click.option('--mask_min_value', help='Mask min value when using face masks', type=float, required=True, default=0.1)
@click.option('--resolution', help='Resolution of output images', type=int, required=True, default=256)
@click.option('--batch_size', help='Batch Size', type=int, required=True, default=4)
@click.option('--learning_rate', help='Learning rate for s estimation, defaults to 2.5',
              type=float, required=True, default=1.5)
@click.option('--clip_gradient_norm', help='clip gradient norm', type=float, required=True, default=0.1)
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
        clip_loss_type: str,
        only_face_mask: int,
        mask_min_value: float,
        resolution: int,
        batch_size: int,
        learning_rate: float,
        clip_gradient_norm: float,
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

    for p in G.parameters():
        p.requires_grad = True
    mean, std = get_mean_std(device)
    img_size = 224
    transf = Compose([Resize(img_size, interpolation=Image.BICUBIC), CenterCrop(img_size)])

    styles_array = torch.tensor(np.load(s_input)['s'], device=device)
    n_items = styles_array.size(0)
    temp_shapes = get_temp_shapes(G)
    resolution_dict = {256: 6, 512: 7, 1024: 8}

    # trainable delta-s
    styles_direction = torch.zeros(1, N_STYLE_CHANNELS, 512, device=device)
    trainable_delta_s = styles_direction.index_select(1, torch.tensor(S_TRAINABLE_SPACE_CHANNELS, device=device))
    trainable_delta_s.requires_grad = True

    checkpoint = torch.load("mobilenet_224_model_best_gdconv_external.pth.tar", map_location=device)
    mobilenet = torch.nn.DataParallel(MobileNet_GDConv(136)).to(device)
    mobilenet.load_state_dict(checkpoint['state_dict'])
    mobilenet.eval()

    landmarks_loss = WingLoss(omega=5)
    id_loss = IDLoss("a").to(device).eval()
    clip_loss1_func, clip_loss2_func = init_clip_loss(clip_loss_type, clip_type, device, text_prompt, negative_text_prompt)

    # if only_face_mask:
    #     idx = np.load(s_input)['idx']
    #     masks_dir = "/".join(s_input.split("/")[:-1]) + "/parsings/"
    #     masks_paths = [f"{masks_dir}/proj{idx[i]:02d}.png" for i in range(n_items)]
    #     masks = [read_image_mask(path, mask_min_value=mask_min_value, dilation=True) for path in masks_paths]

    temp_photos = []
    grads = []
    for i in range(math.ceil(n_items / batch_size)):
        # WARMING UP STEP
        # print(i*batch_size, "processed", time.time()-t1)

        styles = styles_array[i * batch_size:(i + 1) * batch_size].to(device)

        _, img2 = generate_image(G, resolution_dict[resolution], styles, temp_shapes, noise_mode)
        img2_cpu = img2.detach().cpu().numpy()
        temp_photos.append(img2_cpu)

    opt = SGD([trainable_delta_s], lr=learning_rate)
    num_batches = math.ceil(n_items / batch_size)
    total_num_iterations = num_batches * n_epochs
    cur_iteration = 0
    print(f"Total number of iterations: {total_num_iterations}")

    t1 = time.time()
    for epoch in range(n_epochs):
        for _ in range(num_batches):
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
            original_img = torch.tensor(temp_photos[i]).to(device)

            # ------ COMPUTE LOSS --------
            identity_loss, _ = id_loss(img, original_img)
            identity_loss *= identity_loss_coef

            face_landmarks_loss = compute_landmarks_loss(img, original_img, landmarks_loss, landmarks_loss_coef,
                                                         mobilenet, device, mean, std, img_size)

            clip_alignment_loss = compute_clip_loss(
                img, original_img, clip_loss_type, clip_type, clip_loss_coef, clip_loss1_func, clip_loss2_func, transf,
                mean, std, device, text_prompt, negative_text_prompt
            )

            manipulation_direction_loss = trainable_delta_s.norm(2, dim=-1).mean()
            manipulation_direction_loss = manipulation_direction_loss * l2_reg_coef

            loss = identity_loss + clip_alignment_loss + manipulation_direction_loss + face_landmarks_loss

            # ------ COMPUTE LOSS --------

            opt.zero_grad()
            loss.backward(retain_graph=True)

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_delta_s, max_norm=clip_gradient_norm)
            grads.append(trainable_delta_s.grad.clone())
            opt.step()

            print(f"Iteration {cur_iteration}, img size: {img.size(-1)}, gradient norm: {grad_norm:.4f}, "
                  f"lr: {new_learning_rate:.4f}")
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

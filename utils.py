import re
from typing import List

import cv2
from PIL import Image
import numpy as np
import torch

from torch_utils import misc
from torch_utils.ops import upfirdn2d


def block_forward(self, x, img, ws, shapes, force_fp32=False, fused_modconv=None, **layer_kwargs):
    misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
    w_iter = iter(ws.unbind(dim=1))
    dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
    memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
    if fused_modconv is None:
        with misc.suppress_tracer_warnings():  # this value will be treated as a constant
            fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

    # Input.
    if self.in_channels == 0:
        x = self.const.to(dtype=dtype, memory_format=memory_format)
        x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
    else:
        misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
        x = x.to(dtype=dtype, memory_format=memory_format)

    # Main layers.
    if self.in_channels == 0:
        x = self.conv1(x, next(w_iter)[..., :shapes[0]], fused_modconv=fused_modconv, **layer_kwargs)
    elif self.architecture == 'resnet':
        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
        x = y.add_(x)
    else:
        x = self.conv0(x, next(w_iter)[..., :shapes[0]], fused_modconv=fused_modconv, **layer_kwargs)
        x = self.conv1(x, next(w_iter)[..., :shapes[1]], fused_modconv=fused_modconv, **layer_kwargs)

    # ToRGB.
    if img is not None:
        misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
        img = upfirdn2d.upsample2d(img, self.resample_filter)
    if self.is_last or self.architecture == 'skip':
        y = self.torgb(x, next(w_iter)[..., :shapes[2]], fused_modconv=fused_modconv)
        y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        img = img.add_(y) if img is not None else y

    assert x.dtype == dtype
    assert img is None or img.dtype == torch.float32
    return x, img


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def num_range(s: str) -> List[int]:
    """
    Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.
    """

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


def split_ws(G, ws):
    block_ws = []
    misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
    ws = ws.to(torch.float32)

    w_idx = 0
    for res in G.synthesis.block_resolutions:
        block = getattr(G.synthesis, f'b{res}')
        block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
        w_idx += block.num_conv
    return block_ws


def get_mean_std(device):
    mean = torch.as_tensor((0.48145466, 0.4578275, 0.40821073), dtype=torch.float, device=device)
    std = torch.as_tensor((0.26862954, 0.26130258, 0.27577711), dtype=torch.float, device=device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    return mean, std


def get_temp_shapes(G):
    temp_shapes = []
    for res in G.synthesis.block_resolutions:
        block = getattr(G.synthesis, f'b{res}')
        if res == 4:
            temp_shape = (
                block.conv1.affine.weight.shape[0], block.conv1.affine.weight.shape[0],
                block.torgb.affine.weight.shape[0])
            block.conv1.affine = torch.nn.Identity()
            block.torgb.affine = torch.nn.Identity()

        else:
            temp_shape = (
                block.conv0.affine.weight.shape[0], block.conv1.affine.weight.shape[0],
                block.torgb.affine.weight.shape[0])
            block.conv0.affine = torch.nn.Identity()
            block.conv1.affine = torch.nn.Identity()
            block.torgb.affine = torch.nn.Identity()

        temp_shapes.append(temp_shape)
    return temp_shapes


def get_styles(G, ws, block_ws, device):
    with torch.no_grad():
        styles = torch.zeros(ws.size(0), 26, 512, device=device)
        styles_idx = 0
        temp_shapes = []
        for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
            block = getattr(G.synthesis, f'b{res}')

            # first block
            if res == 4:
                temp_shape = (block.conv1.affine.weight.shape[0], block.conv1.affine.weight.shape[0],
                              block.torgb.affine.weight.shape[0])
                styles[:, :1, :] = torch.cat(
                    [block.conv1.affine(cur_ws[i, :1, :]).unsqueeze(1) for i in range(cur_ws.shape[0])], 0)
                styles[:, 1:2, :] = torch.cat(
                    [block.torgb.affine(cur_ws[i, 1:2, :]).unsqueeze(1) for i in range(cur_ws.shape[0])], 0)

                block.conv1.affine = torch.nn.Identity()
                block.torgb.affine = torch.nn.Identity()
                styles_idx += 2
            else:
                temp_shape = (block.conv0.affine.weight.shape[0], block.conv1.affine.weight.shape[0],
                              block.torgb.affine.weight.shape[0])
                styles[:, styles_idx:styles_idx + 1, :temp_shape[0]] = torch.cat(
                    [block.conv0.affine(cur_ws[i, :1, :]).unsqueeze(0) for i in range(cur_ws.shape[0])], 0)
                styles[:, styles_idx + 1:styles_idx + 2, :temp_shape[1]] = torch.cat(
                    [block.conv1.affine(cur_ws[i, 1:2, :]).unsqueeze(0) for i in range(cur_ws.shape[0])], 0)
                styles[:, styles_idx + 2:styles_idx + 3, :temp_shape[2]] = torch.cat(
                    [block.torgb.affine(cur_ws[i, 2:3, :]).unsqueeze(0) for i in range(cur_ws.shape[0])], 0)

                block.conv0.affine = torch.nn.Identity()
                block.conv1.affine = torch.nn.Identity()
                block.torgb.affine = torch.nn.Identity()
                styles_idx += 3
            temp_shapes.append(temp_shape)
    return styles, temp_shapes


def generate_image(G, until_k, styles, temp_shapes, noise_mode, device,
                   use_blending=False, xs_original=None, masks_dict=None):
    if masks_dict is None:
        masks_dict = {}

    x = img = None
    xs = []
    styles_idx = 0
    for k, res in enumerate(G.synthesis.block_resolutions):
        # infer block by block, skip if higher than resolution
        block = getattr(G.synthesis, f'b{res}')
        if k > until_k:
            continue

        if xs_original is not None:
            assert use_blending

        if res == 4:
            x, img = block_forward(block, x, img, styles[:, styles_idx:styles_idx + 2, :], temp_shapes[k],
                                   noise_mode=noise_mode)
            styles_idx += 2
        else:
            x, img = block_forward(block, x, img, styles[:, styles_idx:styles_idx + 3, :], temp_shapes[k],
                                   noise_mode=noise_mode)
            styles_idx += 3

            # blend earrings from original for male2female case
            # long if in order to generate non-blended target image for the first time
            if (res == 32 and use_blending and xs_original is not None
                    and 'earring_mask' in masks_dict and masks_dict['earring_mask'] is not None):
                blending_mask = torch.tensor(cv2.resize(masks_dict['earring_mask'].astype('float'), (res, res),
                                                        interpolation=cv2.INTER_AREA), device=device).unsqueeze(0)
                x = blending_mask * xs_original[k] + (1 - blending_mask) * x

            # blend bg from original
            if res == 64 and use_blending and xs_original is not None and 'bg_mask' in masks_dict:
                blending_mask = torch.tensor(cv2.resize(masks_dict['bg_mask'].astype('float'), (res, res),
                                                        interpolation=cv2.INTER_AREA), device=device).unsqueeze(0)
                x = blending_mask * xs_original[k] + (1 - blending_mask) * x

            # blend teeth from original
            if res == 64 and use_blending and xs_original is not None and 'teeth_mask' in masks_dict:
                blending_mask = torch.tensor(cv2.resize(masks_dict['teeth_mask'].astype('float'), (res, res),
                                                        interpolation=cv2.INTER_AREA), device=device).unsqueeze(0)
                x = blending_mask * xs_original[k] + (1 - blending_mask) * x

            # # blend mouth from global img
            # if res == 32:
            #   blending_mask = torch.tensor(cv2.resize(mouth_mask.astype('float'), (res, res),
            #                                           interpolation=cv2.INTER_AREA), device=device).unsqueeze(0)
            #   x[1] = blending_mask * x[2] + (1 - blending_mask) * x[1]
        if xs_original is not None:
            assert xs_original[k].shape == x.shape
        xs.append(x)

    return xs, img


def read_image_mask(mask_path, mask_min_value=0.0, dilation=True):
    segm_mask = np.array(Image.open(mask_path))
    mask = ((segm_mask == 0) | (segm_mask == 13) | (segm_mask == 14) | (segm_mask == 8) |
            (segm_mask == 9) | (segm_mask == 15) | (segm_mask == 16) | (segm_mask == 18))
    segm_mask = segm_mask.astype('float')
    segm_mask[mask] = mask_min_value
    segm_mask[~mask] = 1
    if dilation:
        segm_mask = cv2.dilate(segm_mask, np.ones((20, 20)), iterations=3)
    return torch.tensor(segm_mask).float()

# Copyright (c) 2020, Roy Or-El. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# This code is a modification of the main.py file
# from the https://github.com/chenxi116/DeepLabv3.pytorch repository

import argparse
import os
import requests
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

import deeplab
from deeplab_utils import download_file, preprocess_image

CLASSES = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
           'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

resnet_file_spec = dict(file_url='https://drive.google.com/uc?id=1oRGgrI4KNdefbWVpw0rRkEP1gbJIRokM',
                        file_path='deeplab_model/R-101-GN-WS.pth.tar', file_size=178260167,
                        file_md5='aa48cc3d3ba3b7ac357c1489b169eb32')
deeplab_file_spec = dict(file_url='https://drive.google.com/uc?id=1w2XjDywFr2NjuUWaLQDRktH7VwIfuNlY',
                         file_path='deeplab_model/deeplab_model.pth', file_size=464446305,
                         file_md5='8e8345b1b9d95e02780f9bed76cc0293')


class CelebASegmentation(Dataset):
    """Dataset for deeplab segmentation, accepts png images"""

    def __init__(self, root, transform=None, crop_size=None):
        self.root = root
        self.transform = transform
        self.crop_size = crop_size

        self.images = [str(x) for x in Path(root).glob('*.png')]

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _img = _img.resize((513, 513), Image.BILINEAR)
        _img = preprocess_image(_img, flip=False, scale=None, crop=(self.crop_size, self.crop_size))

        if self.transform is not None:
            _img = self.transform(_img)

        return _img

    def __len__(self):
        return len(self.images)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=512, help='segmentation output size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--dataset_root', type=str, help='Folder with images', default='out', required=True)
    args = parser.parse_args()
    return args


def get_model():
    model = getattr(deeplab, 'resnet101')(
        pretrained=True,
        num_classes=len(CLASSES),
        num_groups=32,
        weight_std=True,
        beta=False)

    model = model.cuda()
    model.eval()
    if not os.path.isfile(deeplab_file_spec['file_path']):
        print('Downloading DeeplabV3 Model parameters')
        with requests.Session() as session:
            download_file(session, deeplab_file_spec)

        print('Done!')

    checkpoint = torch.load(deeplab_file_spec['file_path'])
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model.load_state_dict(state_dict)
    return model


def infer_model(model, inputs, resolution=512):
    outputs = model(inputs)
    _, pred = torch.max(outputs, 1)
    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
    mask_pred = Image.fromarray(pred)
    mask_pred = mask_pred.resize((resolution, resolution), Image.NEAREST)
    return mask_pred


def get_bg_mask(model_segm, img_arr, device):
    _img = preprocess_image(Image.fromarray(img_arr).resize((513, 513)), flip=False, scale=None, crop=(513, 513))
    mask = infer_model(model_segm, _img.unsqueeze(0).to(device))
    hair_mask = np.array(mask) == CLASSES.index("hair")
    hair_mask = (cv2.dilate(hair_mask.astype('float'), np.ones((10, 10))) > 0)
    bg_mask = (np.array(mask) == CLASSES.index("background")) | (np.array(mask) == CLASSES.index("cloth"))
    bg_mask = (cv2.erode(bg_mask.astype('float'), np.ones((20, 20))) > 0)
    bg_mask[hair_mask] = 0

    # neck_mask = np.array(mask) == CLASSES.index("neck")
    # neck_mask = (cv2.erode(neck_mask.astype('float'), np.ones((7, 7))) > 0)
    #
    # # 'l_brow', 'r_brow', 'hair', 'l_ear', 'r_ear', 'l_eye', 'r_eye'
    # temples_and_lobe_mask = np.array(mask) == CLASSES.index("skin")

    return bg_mask


def get_earring_mouth_lips_masks(model_segm, img_arr, device):
    _img = preprocess_image(Image.fromarray(img_arr).resize((513, 513)), flip=False, scale=None, crop=(513, 513))
    mask = infer_model(model_segm, _img.unsqueeze(0).to(device))
    earring_mask = np.array(mask) == CLASSES.index("ear_r")
    earring_mask = (cv2.dilate(earring_mask.astype('float'), np.ones((15, 15))) > 0)

    mouth_mask = np.array(mask) == CLASSES.index("mouth")
    lips_mask = (np.array(mask) == CLASSES.index("u_lip")) | (np.array(mask) == CLASSES.index("l_lip"))
    teeth_mask = (cv2.erode(mouth_mask.astype('float'), np.ones((5, 5))) > 0)
    mouth_mask = (cv2.dilate((mouth_mask | lips_mask).astype('float'), np.ones((7, 7))) > 0)

    return earring_mask, mouth_mask, teeth_mask


def main():
    args = parse_args()
    resolution = args.resolution
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    dataset_root = args.dataset_root
    assert os.path.isdir(dataset_root)
    dataset = CelebASegmentation(dataset_root, crop_size=513)
    print(f"loaded {len(dataset)} images")

    if not os.path.isfile(resnet_file_spec['file_path']):
        print('Downloading backbone Resnet Model parameters')
        with requests.Session() as session:
            download_file(session, resnet_file_spec)

        print('Done!')

    for i in range(len(dataset)):
        inputs = dataset[i]
        inputs = inputs.cuda()
        model = get_model()
        mask_pred = infer_model(model, inputs.unsqueeze(0), resolution)

        imname = os.path.basename(dataset.images[i])
        try:
            mask_pred.save(dataset.images[i].replace(imname, 'parsings/' + imname[:-4] + '.png'))
        except FileNotFoundError:
            os.makedirs(os.path.join(os.path.dirname(dataset.images[i]), 'parsings'))
            mask_pred.save(dataset.images[i].replace(imname, 'parsings/' + imname[:-4] + '.png'))

        print('processed {0}/{1} images'.format(i + 1, len(dataset)))


if __name__ == "__main__":
    main()

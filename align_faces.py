"""
collection of functions for aligning images using ffhq-style function
"""

import torch
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop
import scipy.ndimage

from mobilenet_facial import MobileNet_GDConv
from utils import get_mean_std
from MTCNN import detect_faces
from warp_images import crop_face


def load_model(device):
    checkpoint = torch.load("mobilenet_224_model_best_gdconv_external.pth.tar", map_location=device)
    mobilenet = torch.nn.DataParallel(MobileNet_GDConv(136)).to(device)
    mobilenet.load_state_dict(checkpoint['state_dict'])
    mobilenet.eval()
    mean, std = get_mean_std(device)
    return mobilenet, mean, std


def detect_landmarks(img, model, device, mean, std):
    """
    Detect landmarks for 1 numpy image
    if no faces are found, return None
    if more than 1 faces are found, return with highest confidence
    """
    img_ = torch.from_numpy(img.copy())
    faces, _ = detect_faces(img_, device=device)
    out_size = 224
    images_cropped = []
    orig_face_metas = []
    if len(faces):
        # torch.Tensor -> torch.Tensor
        cropped, orig_face_size, orig_bbox = crop_face(Image.fromarray(img), [faces[np.argmax(faces[:, 4])]],
                                                       out_size)

        cropped_img = torch.from_numpy(cropped).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
        cropped_img = (cropped_img / 255 - mean.unsqueeze(0)) / std.unsqueeze(0)
        images_cropped.append(cropped_img)
        orig_face_metas.append((orig_face_size, orig_bbox))
    else:
        return None

    img_batch = torch.cat(images_cropped, dim=0)
    with torch.no_grad():
        landmarks = model(img_batch)
    landmarks = landmarks.view(landmarks.size(0), -1, 2)

    for i, (orig_face_size, orig_bbox) in enumerate(orig_face_metas):
        landmarks[i] = landmarks[i] * orig_face_size + torch.tensor([orig_bbox[0], orig_bbox[1]],
                                                                    device=device).view(1, 2)
    landmarks = landmarks.cpu().numpy().astype(int)
    return landmarks[0]


def align_face(image, landmarks, output_size=1024, transform_size=4096, enable_padding=True, rotate_level=True,
               random_shift=0.0, retry_crops=False):
    lm = landmarks.copy()
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    if rotate_level:
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c0 = eye_avg + eye_to_mouth * 0.1
    else:
        x = np.array([1, 0], dtype=np.float64)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c0 = eye_avg + eye_to_mouth * 0.1

    img = Image.fromarray(image)
    quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
    qsize = np.hypot(*x) * 2

    # Keep drawing new random crop offsets until we find one that is contained in the image
    # and does not require padding
    if random_shift != 0:
        for _ in range(1000):
            # Offset the crop rectange center by a random shift proportional to image dimension
            # and the requested standard deviation
            c = (c0 + np.hypot(*x) * 2 * random_shift * np.random.normal(0, 1, c0.shape))
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
            crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                    int(np.ceil(max(quad[:, 1]))))
            if not retry_crops or not (crop[0] < 0 or crop[1] < 0 or crop[2] >= img.width or crop[3] >= img.height):
                # We're happy with this crop (either it fits within the image, or retries are disabled)
                break
        else:
            # rejected N times, give up and move to next image
            # (does not happen in practice with the FFHQ data)
            print('rejected image')
            return

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)
    return img

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

from torchvision.transforms import Compose, Resize, CenterCrop
from utils import get_mean_std
from mobilenet_facial import MobileNet_GDConv
from MTCNN import detect_faces

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def triangle_indices(points):
    convexhull = cv2.convexHull(points)
    # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
    # cv2.fillConvexPoly(mask, convexhull, 255)
    # face_image_1 = cv2.bitwise_and(img, img, mask=mask)
    # Delaunay triangulation

    landmarks_points = points.astype(np.int32)
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
    return indexes_triangles


def get_triangle_points(img, input_points, triangle_idx):
    tr1_pt1 = input_points[triangle_idx[0]]
    tr1_pt2 = input_points[triangle_idx[1]]
    tr1_pt3 = input_points[triangle_idx[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

    rect1 = cv2.boundingRect(triangle1)
    (x, y, w, h) = rect1
    cropped_triangle = img[y: y + h, x: x + w]
    cropped_tr1_mask = np.zeros((h, w), np.uint8)
    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
    return points, cropped_triangle, cropped_tr1_mask


def crop_face(img, faces, out_size):
    if isinstance(img, torch.Tensor):
        width, height = img.size(1), img.size(0)
    else:
        assert isinstance(img, Image.Image)
        width, height = img.size

    x1, y1, x2, y2 = faces[0][:4]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(min([w, h]) * 1.2)
    cx = x1 + w // 2
    cy = y1 + h // 2
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)
    new_bbox = list(map(int, [x1, y1, x2, y2]))

    if isinstance(img, torch.Tensor):
        cropped = img[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]
        cropped_face = F.interpolate(cropped.unsqueeze(0).permute(0, 3, 1, 2),
                                     size=out_size, mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
    else:
        cropped = np.array(img)[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]

        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
        cropped_face = cv2.resize(cropped, (out_size, out_size))
    return cropped_face, cropped.shape[0], new_bbox


if __name__ == "__main__":
    run_path = list(Path("runs").iterdir())[-4]

    checkpoint = torch.load("mobilenet_224_model_best_gdconv_external.pth.tar", map_location=device)
    mobilenet = torch.nn.DataParallel(MobileNet_GDConv(136)).to(device)
    mobilenet.eval()
    mobilenet.load_state_dict(checkpoint['state_dict'])

    mean, std = get_mean_std(device)
    img_size = 224
    transf = Compose([Resize(img_size, interpolation=Image.BICUBIC), CenterCrop(img_size)])

    paths = list(Path(run_path).glob("*.jpeg"))
    for img_path in [paths[0]]:
        img = np.array(Image.open(img_path))

        img1, img2 = Image.fromarray(img[:, :512]), Image.fromarray(img[:, 512:])
        faces1, _ = detect_faces(img1)
        faces2, _ = detect_faces(img2)
        width, height = img1.size
        out_size = 224
        cropped1, orig_face_size1, orig_bbox1 = crop_face(img1, faces1, out_size)
        cropped2, orig_face_size2, orig_bbox2 = crop_face(img2, faces2, out_size)

        img1, img2 = torch.tensor(cropped1).float(), torch.tensor(cropped2).float()
        img_batch = torch.stack([img1, img2]).permute(0, 3, 1, 2).to(device)
        img_batch = (img_batch / 255 - mean.unsqueeze(0)) / std.unsqueeze(0)
        with torch.no_grad():
            landmarks = mobilenet(img_batch)
            landmarks = landmarks.view(landmarks.size(0), -1, 2)
        landmarks_points1 = (landmarks.cpu().numpy()[0] * orig_face_size1 +
                             np.array([orig_bbox1[0], orig_bbox1[1]])[None, :]).astype(int)
        landmarks_points2 = (landmarks.cpu().numpy()[1] * orig_face_size2 +
                             np.array([orig_bbox2[0], orig_bbox2[1]])[None, :]).astype(int)

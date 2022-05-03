import torch
import torch.nn.functional as F
import torch.nn as nn


class LandmarksLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, landmarks1, landmarks2):
        """
        landmarks1: (batch_size, num_landmarks, 2)
        :return:
        """
        landmarks1 = landmarks1[:, 17:].reshape(-1, 2)
        landmarks2 = landmarks2[:, 17:].reshape(-1, 2)
        loss = F.mse_loss(landmarks1, landmarks2)
        return loss

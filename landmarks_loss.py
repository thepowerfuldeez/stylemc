import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import torch
from torch import nn


# torch.log  and math.log is e based
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, landmarks1, landmarks2):
        # except jaw line
        # y = landmarks1[:, 17:].reshape(-1, 2)
        # y_hat = landmarks2[:, 17:].reshape(-1, 2)

        # only lips
        y = landmarks1[:, 48:].reshape(-1, 2)
        y_hat = landmarks2[:, 48:].reshape(-1, 2)

        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


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

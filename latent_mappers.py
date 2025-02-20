import torch
import torchvision.transforms as transforms
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential

from encoder4editing.models.stylegan2.model import EqualLinear, PixelNorm


class ModulationModule(Module):
    def __init__(self, layernum, neg_slope=0.01):
        super(ModulationModule, self).__init__()
        self.layernum = layernum
        self.fc = Linear(512, 512)
        self.norm = LayerNorm([self.layernum, 512], elementwise_affine=False)
        # self.gamma_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512))
        # self.beta_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512))
        self.leakyrelu = LeakyReLU(negative_slope=neg_slope)

    def forward(self, x, embedding):
        x = self.fc(x)
        x = self.norm(x)
        if embedding is not None:
            gamma = self.gamma_function(embedding.float())
            beta = self.beta_function(embedding.float())
            out = x * (1 + gamma) + beta
        else:
            out = x
        out = self.leakyrelu(out)
        return out


class SubMapperModulation(Module):
    def __init__(self, layernum=4, neg_slope=0.01):
        super(SubMapperModulation, self).__init__()
        self.layernum = layernum
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModule(self.layernum, neg_slope) for i in range(5)])

    def forward(self, x, embedding=None):
        x = self.pixelnorm(x)
        for modulation_module in self.modulation_module_list:
            x = modulation_module(x, embedding)
        return x


class SubMapper(Module):
    def __init__(self, latent_dim=512):
        super(SubMapper, self).__init__()
        layers = [PixelNorm()]

        for i in range(4):
            layers.append(
                EqualLinear(latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu')
            )
        self.mapping = Sequential(*layers)

    def forward(self, x):
        bs, nc = x.size(0), x.size(1)
        x = x.reshape(bs * nc, x.size(-1))
        out = self.mapping(x)
        out = out.reshape(bs, nc, out.size(-1))
        return out


class Mapper(Module):
    def __init__(self, neg_slope=0.01):
        super(Mapper, self).__init__()
        self.course_mapping = SubMapperModulation(neg_slope=neg_slope)
        self.medium_mapping = SubMapperModulation(neg_slope=neg_slope)

    def forward(self, x, embedding=None):
        """
        :param x: (batch_size, 8, 512) – style space embedding
        :param clip_embedding:
        :return:
        """

        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]

        if embedding is not None:
            embedding = embedding.unsqueeze(1).repeat(1, 8, 1)
        else:
            embedding = torch.ones(x.size(0), 8, 512, device=x.device)

        x_coarse = self.course_mapping(x_coarse)#, embedding[:, :4, :])
        x_medium = self.medium_mapping(x_medium)#, embedding[:, 4:8, :])

        out = torch.cat([x_coarse, x_medium], dim=1)
        return out

import torch
import torchvision.transforms as transforms
from torch import nn
from torch.nn import Module
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential

from encoder4editing.models.stylegan2.model import EqualLinear, PixelNorm


class ModulationModule(Module):
    def __init__(self, layernum):
        super(ModulationModule, self).__init__()
        self.layernum = layernum
        self.fc = Linear(512, 512)
        self.norm = LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.gamma_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512))
        self.beta_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512))
        self.leakyrelu = LeakyReLU()

    def forward(self, x, embedding):
        x = self.fc(x)
        x = self.norm(x)
        if embedding is not None:
            gamma = self.gamma_function(embedding.float())
            beta = self.beta_function(embedding.float())
            out = x * (1 + gamma) + beta
        out = self.leakyrelu(out)
        return out


class SubMapper(Module):
    def __init__(self, layernum):
        super(SubMapper, self).__init__()
        self.layernum = layernum
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModule(self.layernum) for i in range(5)])

    def forward(self, x, embedding=None):
        x = self.pixelnorm(x)
        for modulation_module in self.modulation_module_list:
            x = modulation_module(x, embedding)
        return x


class Mapper(Module):
    def __init__(self):
        super(Mapper, self).__init__()
        self.course_mapping = SubMapper(4)
        self.medium_mapping = SubMapper(4)

    def forward(self, x, clip_embedding=None):
        """
        :param x: (batch_size, 8, 512) – style space embedding
        :param clip_embedding:
        :return:
        """

        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]

        x_coarse = self.course_mapping(x_coarse)#, clip_embedding[:, :4, :])
        x_medium = self.medium_mapping(x_medium)#, clip_embedding[:, 4:8, :])

        out = torch.cat([x_coarse, x_medium], dim=1)
        return out

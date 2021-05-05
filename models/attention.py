import torch
import torch.nn.functional as F
from torch import nn


class Unary(nn.Module):
    def __init__(self, input_embed_channels, output_embed_channels, kernel_size=(1, 1), stride=(1, 1),
                 padding=(0, 0), name='unary_att'):
        super(Unary, self).__init__()
        self.w1 = nn.Conv2d(in_channels=input_embed_channels,
                            out_channels=output_embed_channels // 2,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,)
        self.w2 = nn.Conv2d(in_channels=output_embed_channels // 2,
                            out_channels=output_embed_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.w3 = nn.Conv2d(in_channels=output_embed_channels,
                            out_channels=1,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        # self.dropout = nn.Dropout2d(dropout_prob)
        self.w4 = nn.Conv2d(in_channels=1,
                            out_channels=1,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.name = name

    def forward(self, x):
        x = self.w1(x)
        x = F.normalize(x)
        x = torch.relu(x)
        x = self.w2(x)
        x = F.normalize(x)
        x = torch.relu(x)
        x = self.w3(x)
        x = self.w4(x)
        # x = F.normalize(x)
        # 1 X 16 X 16

        return x.squeeze(1)


class SelfAttn(nn.Module):
    def __init__(self, input_embed_channels, output_embed_channels, kernel_size=(1, 1), name='self_attn', learnable=True):
        super(SelfAttn, self).__init__()
        self.learnable = learnable
        if self.learnable:
            self.unary = Unary(input_embed_channels=input_embed_channels,
                               output_embed_channels=output_embed_channels,
                               kernel_size=kernel_size)
            self.name = name

    def forward(self, x):
        tau = self.unary(x)
        tau = torch.softmax(tau.view(tau.shape[0], -1), dim=1).reshape(tau.shape)
        attended_x = torch.einsum('bcxy,bxy->bcxy', x, tau)
        return attended_x, tau
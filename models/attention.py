import torch
import torch.nn.functional as F
from torch import nn


class SumPool(nn.Module):
    def __init__(self, factor=2):
        super(SumPool, self).__init__()
        self.factor = factor
        self.avgpool = nn.AvgPool2d(kernel_size=(factor, factor), stride=(factor, factor), padding=(0, 0))

    def forward(self, x):
        return self.avgpool(x) * (self.factor ** 2)


class SimpleSelfAttention(nn.Module):
    def __init__(self, input_channels, embed_channels, kernel_size=(1, 1), stride=(1, 1),
                 padding=(0, 0), name='simple_self_attn'):
        super(SimpleSelfAttention, self).__init__()
        self.w1 = nn.Conv2d(in_channels=input_channels,
                            out_channels=embed_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,)
        self.w2 = nn.Conv2d(in_channels=embed_channels,
                            out_channels=1,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.name = name

    def forward(self, x):
        tau = self.w1(x)
        tau = F.normalize(tau)
        tau = self.relu(tau)
        tau = self.w2(tau).squeeze(1)
        tau = torch.softmax(tau.flatten(1), dim=1).reshape(tau.shape)
        attended_x = torch.einsum('bcxy,bxy->bcxy', x, tau)
        return attended_x, tau


class SimpleUnary(nn.Module):
    def __init__(self, input_embed_channels, output_embed_channels, kernel_size=(1, 1), stride=(1, 1),
                 padding=(0, 0), name='unary_att'):
        super(SimpleUnary, self).__init__()
        self.w1 = nn.Conv2d(in_channels=input_embed_channels,
                            out_channels=output_embed_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,)
        self.w2 = nn.Conv2d(in_channels=output_embed_channels,
                            out_channels=1,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        # self.dropout = nn.Dropout2d(dropout_prob)
        self.name = name

    def forward(self, x):
        x = self.w1(x)
        x = F.normalize(x)
        x = torch.relu(x)
        x = self.w3(x)
        # x = F.normalize(x)
        # 1 X 16 X 16

        return x.squeeze(1)


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


class Marginals(nn.Module):
    def __init__(self, margin_dim=256, factor=2):
        super(Marginals, self).__init__()
        self.factor = factor
        self.margin_dim = margin_dim
        self.tau_pool = SumPool(factor=factor)
        self.lamb = nn.Parameter(torch.ones(1, 1, margin_dim, margin_dim))
        self.name = "marginals"

    def forward(self, tau1, tau2):
        tau1_marginal = self.tau_pool(tau1)
        tau1_lamb = tau1_marginal * torch.exp(-self.lamb)
        tau1_lamb_sum = tau1_lamb.view(tau1_lamb.shape[0], -1).sum(dim=1)
        tau1_lamb = tau1_lamb / tau1_lamb_sum
        tau2_lamb = tau2 * torch.exp(self.lamb)
        tau2_lamb_sum = tau2_lamb.view(tau2_lamb.shape[0], -1).sum(dim=1)
        tau2_lamb = tau2_lamb / tau2_lamb_sum
        return (tau1, tau2), (tau1_lamb, tau2_lamb)


class MarginalsExtended(nn.Module):
    def __init__(self, margin_dim=256, factor=2):
        super(MarginalsExtended, self).__init__()
        self.factor = factor
        self.margin_dim = margin_dim
        self.tau_pool = SumPool(factor=factor)
        self.lamb = nn.Parameter(torch.ones(1, 1, margin_dim, margin_dim))
        self.name = "marginals_extended"

    def forward(self, tau1, tau2, tau3, tau4):

        lamb2 = SumPool(2)(self.lamb)
        lamb3 = SumPool(4)(self.lamb)
        lamb4 = SumPool(8)(self.lamb)

        tau1_lamb_neg = SumPool(2)(tau1) * torch.exp(-lamb2)
        tau1_lamb_neg = tau1_lamb_neg / tau1_lamb_neg.view(tau1_lamb_neg.shape[0], -1).sum(dim=1)

        tau2_lamb = tau2 * torch.exp(lamb2)
        tau2_lamb = tau2_lamb / tau2_lamb.view(tau2_lamb.shape[0], -1).sum(dim=1)
        tau2_lamb_neg = SumPool(2)(tau2) * torch.exp(-lamb3)
        tau2_lamb_neg = tau2_lamb_neg / tau2_lamb_neg.view(tau2_lamb_neg.shape[0], -1).sum(dim=1)
        
        tau3_lamb = tau3 * torch.exp(lamb3)
        tau3_lamb = tau3_lamb / tau3_lamb.view(tau3_lamb.shape[0], -1).sum(dim=1)
        tau3_lamb_neg = SumPool(2)(tau3) * torch.exp(-lamb4)
        tau3_lamb_neg = tau3_lamb_neg / tau3_lamb_neg.view(tau3_lamb_neg.shape[0], -1).sum(dim=1)

        tau4_lamb = tau4 * torch.exp(lamb4)
        tau4_lamb = tau4_lamb / tau4_lamb.view(tau4_lamb.shape[0], -1).sum(dim=1)
        # source, target
        return (tau1_lamb_neg, tau2_lamb), (tau2_lamb_neg, tau3_lamb), (tau3_lamb_neg, tau4_lamb)


if __name__ == "__main__":

    layer = MarginalsExtended(margin_dim=256)
    init_size = 256
    taus = [torch.randn(1, 1, 256 // (2**i), 256 // (2**i)) for i in range(4)]
    res = layer(*taus)
    pass
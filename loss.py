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


def attn_kl_div(input, target, detach_target=False):
    assert len(input.shape) == 3 and len(target.shape) == 3, "Expected 3D tensors"
    assert input.shape[-1] == input.shape[-2] and target.shape[-1] == target.shape[-2], "Spatial dimensions must be equal"
    assert input.shape[-1] > target.shape[-1]
    factor = input.shape[-1] // target.shape[-1]
    input_marginal = (nn.AvgPool2d(kernel_size=(factor, factor), stride=(factor, factor), padding=(0, 0))(input.unsqueeze(1)) * factor**2).squeeze(1)
    if detach_target:
        target = target.detach()
    kl = F.kl_div(target.log(), input_marginal, reduction='batchmean')
    return kl


class MarginalPenaltyLoss(nn.Module):
    def __init__(self, attn_kl=True, kl_weight=1e-4, detach_targets=False):
        super(MarginalPenaltyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.attn_kl = attn_kl
        self.kl_weight = kl_weight
        self.detach_targets = detach_targets

    def forward(self, output, targets, attn):
        cross_entropy_loss = self.cross_entropy(output, targets)
        if self.attn_kl and attn is not None:
            if len(attn) == 4:
                tau3_lamb, tau4_lamb, tau3, tau4 = attn
                if self.detach_targets:
                    tau4_lamb = tau4_lamb.detach()
                kl_loss = F.kl_div(tau3_lamb.log(), tau4_lamb, reduction='batchmean')
                total_loss = cross_entropy_loss + self.kl_weight * kl_loss
                return (cross_entropy_loss, kl_loss, total_loss)
            else:
                raise NotImplementedError
        return (cross_entropy_loss, )


class MarginalsPenaltyLossExtended(nn.Module):
    def __init__(self, init_dim=256, num_tau=4, pool_factor=2, attn_kl=True, kl_weight=1.0, detach_targets=True):
        super(MarginalsPenaltyLossExtended, self).__init__()
        self.init_dim = init_dim
        self.num_tau = num_tau
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.attn_kl = attn_kl
        self.kl_weight = kl_weight
        self.detach_targets = detach_targets
        self.tau_pool = SumPool(factor=pool_factor)
        lamb_sizes = [self.init_dim // 2**i for i in range(self.num_tau)]
        self.lamb_params = [nn.Parameter(torch.ones(1, 1, d, d)) for d in lamb_sizes]

    def forward(self, output, targets, taus):
        cross_entropy_loss = self.cross_entropy(output, targets)
        if self.attn_kl:
            marginals = [self.tau_pool(tau) for tau in taus[:self.num_tau-1]]
            minus_mult_marginals = [m * torch.exp(-self.lamb_params[i+1]) for i, m in enumerate(marginals)]
            minus_mult_marginals_sums = [m.view(m.shape[0], -1).sum(dim=1) for m in minus_mult_marginals]
            minus_mult_marginals = [minus_mult_marginals[i] / minus_mult_marginals_sums[i] for i in range(len(minus_mult_marginals))]

            plus_mult_taus = [t * torch.exp(self.lamb_params[i]) for i, t in enumerate(taus)]
            plus_mult_taus_sums = [m.view(m.shape[0], -1).sum(dim=1) for m in plus_mult_taus]
            plus_mult_taus = [plus_mult_taus[i] / plus_mult_taus_sums[i] for i in range(len(plus_mult_taus))]

            tau_marginal_pairs = [(plus_mult_taus[i+1], minus_mult_marginals[i]) for i in range(len(minus_mult_marginals))]

            kl_losses = []
            for tau, marginal in tau_marginal_pairs:
                if self.detach_targets:
                    tau = tau.detach()
                kl_losses.append(F.kl_div(marginal.log(), tau, reduction='batchmean'))

            total_loss = cross_entropy_loss + self.kl_weight * sum(kl_losses)
            return cross_entropy_loss, kl_losses, total_loss
        return (cross_entropy_loss, )


class TauKLDivLoss(nn.Module):
    def __init__(self, attn_kl=True, kl_weight=1e-4, detach_targets=False):
        super(TauKLDivLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.attn_kl = attn_kl
        self.kl_weight = kl_weight
        self.detach_targets = detach_targets

    def forward(self, output, targets, attn):
        cross_entropy_loss = self.cross_entropy(output, targets)
        if self.attn_kl and attn is not None:
            if len(attn) == 2:
                tau3, tau4 = attn
                kl_loss = attn_kl_div(tau3, tau4, detach_target=self.detach_targets)
                total_loss = cross_entropy_loss + self.kl_weight * kl_loss
                return (cross_entropy_loss, kl_loss, total_loss)
            elif len(attn) == 3:
                tau2, tau3, tau4 = attn
                kl23 = attn_kl_div(tau2, tau3, detach_target=self.detach_targets)
                kl34 = attn_kl_div(tau3, tau4, detach_target=self.detach_targets)
                # kl24 = attn_kl_div(tau2, tau4, detach_target=self.detach_targets)
                total_loss = cross_entropy_loss + self.kl_weight * (kl23 + kl34)
                return (cross_entropy_loss, (kl23, kl34, None), total_loss)
            else:
                raise NotImplementedError
        return (cross_entropy_loss, )


if __name__ == "__main__":
    num_tau = 2
    layer = MarginalsPenaltyLossExtended(num_tau=num_tau)
    init_size = 256
    taus = [torch.randn(1, 1, 256 // (2**i), 256 // (2**i)) for i in range(num_tau)]
    layer(taus)
    pass
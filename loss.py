import torch
import torch.nn.functional as F


def attn_kl_div(input, target, detach_target=False):
    assert len(input.shape) == 3 and len(target.shape) == 3, "Expected 3D tensors"
    assert input.shape[-1] == input.shape[-2] and target.shape[-1] == target.shape[-2], "Spatial dimensions must be equal"
    assert input.shape[-1] > target.shape[-1]
    factor = input.shape[-1] // target.shape[-1]
    input_marginal = (torch.nn.AvgPool2d(kernel_size=(factor, factor), stride=(factor, factor), padding=(0, 0))(input.unsqueeze(1)) * factor**2).squeeze(1)
    if detach_target:
        target = target.detach()
    kl = F.kl_div(target.log(), input_marginal, reduction='batchmean')
    return kl


class MarginalPenaltyLoss(torch.nn.Module):
    def __init__(self, attn_kl=True, kl_weight=1e-4):
        super(MarginalPenaltyLoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=self.loss_weights)
        self.attn_kl = attn_kl
        self.kl_weight = kl_weight

    def forward(self, output, targets, attn):
        cross_entropy_loss = self.cross_entropy(output, targets)
        if self.attn_kl and attn is not None:
            if len(attn) == 4:
                tau3_lamb, tau4_lamb, tau3, tau4 = attn
                kl_loss = F.kl_div(tau3_lamb.log(), tau4_lamb, reduction='batchmean')
                total_loss = cross_entropy_loss + self.kl_weight * kl_loss
                return (cross_entropy_loss, kl_loss, total_loss)
            else:
                raise NotImplementedError
        return (cross_entropy_loss, )


class TauKLDivLoss(torch.nn.Module):
    def __init__(self, attn_kl=True, kl_weight=1e-4, detach_targets=False):
        super(TauKLDivLoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
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
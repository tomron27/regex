import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from models.attention import SelfAttn, Marginals


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetRegressor(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        channels: int = 4,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_factor: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        learnable_attn: bool = True,
        **kwargs
    ) -> None:
        super(ResNetRegressor, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.learnable_attn = learnable_attn
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group * width_factor
        self.conv1 = nn.Conv2d(channels, self.inplanes, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        if self.learnable_attn:
            # self.self_attn1 = SelfAttn(input_embed_channels=256, output_embed_channels=2048)
            # self.self_attn2 = SelfAttn(input_embed_channels=512, output_embed_channels=2048)
            self.self_attn3 = SelfAttn(input_embed_channels=1024, output_embed_channels=2048)
            self.self_attn4 = SelfAttn(input_embed_channels=2048, output_embed_channels=2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(512 * block.expansion, 1000)
        self.final = nn.Linear(1000, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                try:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                except AttributeError:
                    pass

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # if self.learnable_attn:
            # x, p1 = self.self_attn1(x)
        x = self.layer2(x)
        # if self.learnable_attn:
            # x, p2 = self.self_attn2(x)
        x = self.layer3(x)
        if self.learnable_attn:
            x, p3 = self.self_attn3(x)
        x = self.layer4(x)
        if self.learnable_attn:
            x, p4 = self.self_attn4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dense(x)
        x = self.final(x).squeeze()

        if self.learnable_attn:
            return x, (p3, p4)
        else:
            return x, (None, None)


class ResNetClassifier(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        channels: int = 4,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_factor: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        num_classes: int = 2,
        learnable_attn: bool = True,
        learnable_marginals: bool = True,
        **kwargs
    ) -> None:
        super(ResNetClassifier, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.learnable_attn = learnable_attn
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group * width_factor
        self.conv1 = nn.Conv2d(channels, self.inplanes, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.learnable_marginals = learnable_marginals
        if self.learnable_marginals:
            self.marginals = Marginals(spatial_dim=8)
        if self.learnable_attn:
            # self.self_attn1 = SelfAttn(input_embed_channels=256, output_embed_channels=2048)
            # self.self_attn2 = SelfAttn(input_embed_channels=512, output_embed_channels=2048)
            self.self_attn3 = SelfAttn(input_embed_channels=1024, output_embed_channels=2048)
            self.self_attn4 = SelfAttn(input_embed_channels=2048, output_embed_channels=2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                try:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                except AttributeError:
                    pass

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # if self.learnable_attn:
            # x, p1 = self.self_attn1(x)
        x = self.layer2(x)
        # if self.learnable_attn:
            # x, p2 = self.self_attn2(x)
        x = self.layer3(x)
        if self.learnable_attn:
            x, p3 = self.self_attn3(x)
        x = self.layer4(x)
        if self.learnable_attn:
            x, p4 = self.self_attn4(x)

        if self.learnable_attn and self.learnable_marginals:
            p3_lamb, p4_lamb, p3, p4 = self.marginals(p3, p4)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        if self.learnable_attn:
            if self.learnable_attn:
                return x, (p3_lamb, p4_lamb, p3, p4)
            return x, (p3, p4)
        else:
            return x, None


def get_resnet50_attn_regressor(**kwargs):
    model = ResNetRegressor(Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    if kwargs["weights"] is not None:
        print("Loading pretrained model from: '{}'".format(kwargs["weights"]))
        weights = torch.load(kwargs["weights"])
        # Remove ImageNet specific weights
        if "imagenet" in kwargs["weights"]:
            weights.pop('conv1.weight')
            weights.pop('fc.weight')
            weights.pop('fc.bias')
        model.load_state_dict(weights, strict=False)
    if kwargs["freeze_backbone"]:
        # Enable training only on the self attention / final layers
        for layer in model.modules():
            for p in layer.parameters():
                p.requires_grad = False
        for layer in model.modules():
            if hasattr(layer, "name") or any([s in layer._get_name() for s in ["Linear"]]):
                for p in layer.parameters():
                    p.requires_grad = True
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters: {}\nTotal trainable parameters: {}".format(total_params, trainable_params))
    return model


def get_resnet50_attn_classifier(**kwargs):
    model = ResNetClassifier(Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    if kwargs["weights"] is not None:
        print("Loading pretrained model from: '{}'".format(kwargs["weights"]))
        weights = torch.load(kwargs["weights"])
        # Remove ImageNet specific weights
        if "imagenet" in kwargs["weights"]:
            weights.pop('conv1.weight')
            weights.pop('fc.weight')
            weights.pop('fc.bias')
        model.load_state_dict(weights, strict=False)
    if kwargs["freeze_backbone"]:
        # Enable training only on the self attention / final layers
        for layer in model.modules():
            for p in layer.parameters():
                p.requires_grad = False
        for layer in model.modules():
            if hasattr(layer, "name") or any([s in layer._get_name() for s in ["Linear"]]):
                for p in layer.parameters():
                    p.requires_grad = True
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters: {}\nTotal trainable parameters: {}".format(total_params, trainable_params))
    return model




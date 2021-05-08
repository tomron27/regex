from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, channels=1, out_channels=1, init_features=32, spatial_dim=240, **kwargs):
        super(UNet, self).__init__()

        self.features = init_features
        self.spatial_dim = spatial_dim
        self.in_channels = channels
        self.out_channels = out_channels

        self.encoder1 = UNet._block(self.in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.encoder2 = UNet._block(self.features, self.features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.encoder3 = UNet._block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.encoder4 = UNet._block(self.features * 4, self.features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.bottleneck = UNet._block(self.features * 8, self.features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            self.features * 16, self.features * 8, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder4 = UNet._block((self.features * 8) * 2, self.features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            self.features * 8, self.features * 4, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder3 = UNet._block((self.features * 4) * 2, self.features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            self.features * 4, self.features * 2, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder2 = UNet._block((self.features * 2) * 2, self.features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            self.features * 2, self.features, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder1 = UNet._block(self.features * 2, self.features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=self.features, out_channels=out_channels, kernel_size=(1, 1)
        )
        self.dense = nn.Linear(self.spatial_dim * self.spatial_dim, 1)
        # self.fc1 = nn.Linear(self.spatial_dim * self.spatial_dim, self.spatial_dim)
        # self.fc2 = nn.Linear(self.spatial_dim, 1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)

        out = dec1.view(-1, self.spatial_dim * self.spatial_dim)
        # out = self.fc1(out)
        # out = self.fc2(out)
        out = self.dense(out).squeeze()
        return out, (None, None)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def get_unet_regressor(**kwargs):
    model = UNet(**kwargs)
    if kwargs["weights"] is not None:
        print("Loading pretrained model from: '{}'".format(kwargs["weights"]))
        weights = torch.load(kwargs["weights"])
        model.load_state_dict(weights, strict=False)
    # if kwargs["freeze_backbone"]: # TODO
    #     # Enable training only on the self attention / final layers
    #     for layer in model.modules():
    #         for p in layer.parameters():
    #             p.requires_grad = False
    #     for layer in model.modules():
    #         if hasattr(layer, "name") or any([s in layer._get_name() for s in ["Linear"]]):
    #             for p in layer.parameters():
    #                 p.requires_grad = True
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters: {}\nTotal trainable parameters: {}".format(total_params, trainable_params))
    return model


# if __name__ == "__main__":
#     model = UNet(init_features=64)
#     print("Num of params:", sum(p.numel() for p in model.parameters()))
#     inputs = torch.randn(64, 1, 240, 240)
#     model(inputs)
import torch.nn.init as init
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F
from torch import nn


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(1, 4)
        self.enc2 = _EncoderBlock(4, 8)
        self.enc3 = _EncoderBlock(8, 16)
        self.enc4 = _EncoderBlock(16, 32, dropout=True)
        self.center = _DecoderBlock(32, 64, 32)
        self.dec4 = _DecoderBlock(64, 32, 16)
        self.dec3 = _DecoderBlock(32, 16, 8)
        self.dec2 = _DecoderBlock(16, 8, 4)
        self.dec1 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(num_classes, 64, kernel_size=1)
        #self.final = nn.Conv2d(num_classes, 64, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        #print(x.shape)
        enc1 = self.enc1(x)
        #print(enc1.shape)
        enc2 = self.enc2(enc1)
        #print(enc2.shape)
        enc3 = self.enc3(enc2)
        #print(enc3.shape)
        enc4 = self.enc4(enc3)
        #print(enc4.shape)
        center = self.center(enc4)
        #print(center.shape)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        #print(dec4.shape)
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        #print(dec3.shape)
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        #print(dec2.shape)
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        #print(dec1.shape)
        final = self.final(dec1)
        #print(f'final : {final.shape}')
        output = F.upsample(final, x.size()[2:], mode='bilinear')
        #print(f'ouput : {output.shape}')

        return output

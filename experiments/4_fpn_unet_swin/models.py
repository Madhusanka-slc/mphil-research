"""
Segmentation Models: U-Net, FPN, Swin-Transformer
Based on Paper 2: Yousefzadeh et al. (2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, max(channels // reduction, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(channels // reduction, 1), channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.shape
        squeeze = self.avg_pool(x).view(batch, channels)
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(batch, channels, 1, 1)
        return x * excitation


class UNetWithSEResNet18(nn.Module):
    """U-Net with SE-ResNet18 encoder."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Load ResNet18 backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify first layer for single channel input
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder blocks with SE
        self.layer1 = self._add_se_to_layer(resnet.layer1)
        self.layer2 = self._add_se_to_layer(resnet.layer2)
        self.layer3 = self._add_se_to_layer(resnet.layer3)
        self.layer4 = self._add_se_to_layer(resnet.layer4)

        # Decoder (upsampling path)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(256)
        )

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(128)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )

        self.final = nn.Conv2d(64, out_channels, 1)

    def _add_se_to_layer(self, layer):
        for block in layer:
            block.se = SEBlock(block.conv2.out_channels)
        return layer

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Decoder
        x = self.up4(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x0], dim=1)
        x = self.dec1(x)

        x = self.final(x)
        return x


class FPNWithSEResNet18(nn.Module):
    """Feature Pyramid Network with SE-ResNet18 encoder."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Load ResNet18 backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify first layer for single channel input
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder layers
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # FPN lateral connections
        self.lat_conv4 = nn.Conv2d(512, 256, 1)
        self.lat_conv3 = nn.Conv2d(256, 256, 1)
        self.lat_conv2 = nn.Conv2d(128, 256, 1)
        self.lat_conv1 = nn.Conv2d(64, 256, 1)

        # FPN output convolutions
        self.fpn_conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn_conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn_conv1 = nn.Conv2d(256, 256, 3, padding=1)

        # Decoder head
        self.decode4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(128)
        )
        self.decode3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(128)
        )
        self.decode2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(128)
        )
        self.decode1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(128)
        )

        # Final segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x)
        x = self.maxpool(c1)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # FPN
        p5 = self.lat_conv4(c5)
        p4 = self.fpn_conv4(p5 + F.interpolate(self.lat_conv3(c4), p5.shape[2:], mode='nearest'))
        p3 = self.fpn_conv3(p4 + F.interpolate(self.lat_conv2(c3), p4.shape[2:], mode='nearest'))
        p2 = self.fpn_conv2(p3 + F.interpolate(self.lat_conv1(c2), p3.shape[2:], mode='nearest'))
        p1 = self.fpn_conv1(F.interpolate(p2, c1.shape[2:], mode='nearest'))

        # Decoder
        d4 = self.decode4(F.interpolate(p5, c4.shape[2:], mode='nearest'))
        d3 = self.decode3(F.interpolate(p4, c3.shape[2:], mode='nearest'))
        d2 = self.decode2(F.interpolate(p3, c2.shape[2:], mode='nearest'))
        d1 = self.decode1(p2)

        # Concatenate all pyramid levels
        d = torch.cat([
            F.interpolate(d4, d2.shape[2:], mode='nearest'),
            F.interpolate(d3, d2.shape[2:], mode='nearest'),
            d2,
            F.interpolate(d1, d2.shape[2:], mode='nearest')
        ], dim=1)

        # Final segmentation
        out = self.segmentation_head(d)
        out = F.interpolate(out, size=(x.shape[2]*4, x.shape[3]*4), mode='bilinear', align_corners=False)
        return out


class SwinTransformerSegmentation(nn.Module):
    """Swin-Transformer for segmentation."""
    def __init__(self, in_channels=1, out_channels=1, image_size=384):
        super().__init__()

        self.image_size = image_size

        # Load Swin-Transformer backbone
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True)

        # Modify for single channel input
        self.backbone.patch_embed.proj = nn.Conv2d(
            in_channels,
            self.backbone.patch_embed.proj.out_channels,
            kernel_size=4,
            stride=4
        )

        # Get backbone output channels
        self.backbone_out_channels = self.backbone.num_features

        # Decoder head
        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x):
        # Backbone
        x = self.backbone.forward_features(x)

        # Reshape if needed
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, self.backbone_out_channels, -1, -1)

        # Decoder
        x = self.decoder(x)
        return x


def create_model(model_name, in_channels=1, out_channels=1, image_size=384):
    """Factory function to create models."""
    if model_name == 'unet':
        return UNetWithSEResNet18(in_channels, out_channels)
    elif model_name == 'fpn':
        return FPNWithSEResNet18(in_channels, out_channels)
    elif model_name == 'swin':
        return SwinTransformerSegmentation(in_channels, out_channels, image_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == '__main__':
    import torch

    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(1, 1, 384, 384).to(device)

    for model_name in ['unet', 'fpn', 'swin']:
        print(f"\nTesting {model_name}...")
        model = create_model(model_name).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {params:,}")

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class Router(nn.Module):
    def __init__(self, in_ch: int, n_experts: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_ch // 2, n_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.mlp(x), dim=-1)


class ConvExpert(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        mid = in_ch // 2
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, mid, 1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, mid // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid // 4, mid, 1),
            nn.Sigmoid()
        )

        self.out = nn.Conv2d(mid, in_ch, 1)
        self.bn2 = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        y = self.act(self.bn1(self.pw(self.dw(x))))
        y = y * self.se(y)
        y = self.bn2(self.out(y))
        return self.act(y + x)


class FPNDecoder(nn.Module):
    def __init__(self, in_channels: List[int], out_ch: int = 256):
        super().__init__()

        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_ch, 1) for c in in_channels
        ])

        self.smooth = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in in_channels
        ])

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        feats = list(feats)

        p = self.lateral[-1](feats[-1])
        outputs = [p]
        for idx in range(3, 0, -1):
            up = F.interpolate(p, size=feats[idx - 1].shape[-2:], mode="nearest")
            p = self.lateral[idx - 1](feats[idx - 1]) + up
            outputs.insert(0, p)

        outputs = [self.smooth[i](p) for i, p in enumerate(outputs)]

        p2_sz = outputs[0].shape[-2:]
        fused = sum(F.interpolate(p, size=p2_sz, mode="nearest") for p in outputs)
        return fused


def create_encoder(encoder_name, encoder_weights='imagenet'):
    """Create encoder from timm with error handling and fallbacks"""

    try:
        encoder = timm.create_model(
            encoder_name,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            pretrained=(encoder_weights == 'imagenet')
        )
        return encoder, encoder.feature_info.channels()
    except Exception as e:
        print(f"Warning: Failed to create encoder '{encoder_name}': {e}")

        fallback_encoders = [
            'resnext50_32x4d',
            'resnet50',
            'resnet34',
            'efficientnet_b2',
            'efficientnet_b0'
        ]

        for fallback in fallback_encoders:
            if fallback != encoder_name:
                try:
                    print(f"Trying fallback encoder: {fallback}")
                    encoder = timm.create_model(
                        fallback,
                        features_only=True,
                        out_indices=(1, 2, 3, 4),
                        pretrained=(encoder_weights == 'imagenet')
                    )
                    return encoder, encoder.feature_info.channels()
                except Exception as fe:
                    print(f"Fallback {fallback} also failed: {fe}")
                    continue

        raise RuntimeError(f"Could not create any encoder. Original error: {e}")


class ImprovedMultiTaskMoENet(nn.Module):
    def __init__(self,
                 backbone_name: str = "resnext50_32x4d",
                 seg_classes: int = 3,
                 cls_classes: int = 5,
                 n_experts: int = 8,
                 decoder_channels: int = 256,
                 encoder_weights: str = 'imagenet'):
        super().__init__()

        self.backbone, chans = create_encoder(backbone_name, encoder_weights)

        self.router = Router(chans[-1], n_experts)
        self.experts = nn.ModuleList([ConvExpert(chans[-1]) for _ in range(n_experts)])

        self.decoder = FPNDecoder(chans, out_ch=decoder_channels)

        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels // 2, decoder_channels // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels // 2, seg_classes, 1)
        )

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(chans[-1], chans[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(chans[-1] // 2, cls_classes)
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        c5 = feats[-1]

        weights = self.router(c5)
        exp_out = torch.stack([e(c5) for e in self.experts], dim=1)
        c5_fused = torch.sum(weights[:, :, None, None, None] * exp_out, dim=1)
        feats[-1] = c5_fused

        p2 = self.decoder(feats)
        seg_logits = self.seg_head(p2)
        seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode='bilinear', align_corners=False)

        cls_logits = self.cls_head(c5_fused)

        return {"seg": seg_logits, "cls": cls_logits, "router_weights": weights}


def create_model(config):
    """Create model based on config"""
    model = ImprovedMultiTaskMoENet(
        backbone_name=getattr(config.model, 'backbone_name', 'resnext50_32x4d'),
        seg_classes=config.model.seg_classes,
        cls_classes=config.model.cls_classes,
        n_experts=getattr(config.model, 'n_experts', 8),
        decoder_channels=getattr(config.model, 'decoder_channels', 256),
        encoder_weights=getattr(config.model, 'encoder_weights', 'imagenet')
    )
    return model
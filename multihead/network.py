import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ClassificationHead(nn.Module):
    def __init__(self, encoder_channels, num_classes=5):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(encoder_channels, num_classes)

    def forward(self, x):
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name="resnext50_32x4d", encoder_weights="imagenet",
                 seg_classes=3, cls_classes=5, segmentation_head="Segformer"):
        super().__init__()

        # Create the segmentation model first
        if segmentation_head.lower() == "segformer":
            seg_model = smp.Segformer(
                encoder_name = encoder_name,
                encoder_weights = encoder_weights,
                in_channels = 3,
                classes = seg_classes,
                activation = None
            )
        elif segmentation_head.lower() == "unet":
            seg_model = smp.Unet(
                encoder_name = encoder_name,
                encoder_weights = encoder_weights,
                in_channels = 3,
                classes = seg_classes,
                activation = None
            )
        elif segmentation_head.lower() == "deeplabv3plus":
            seg_model = smp.DeepLabV3Plus(
                encoder_name = encoder_name,
                encoder_weights = encoder_weights,
                in_channels = 3,
                classes = seg_classes,
                activation = None
            )
        else:
            # Default to Segformer
            seg_model = smp.Segformer(
                encoder_name = encoder_name,
                encoder_weights = encoder_weights,
                in_channels = 3,
                classes = seg_classes,
                activation = None
            )

        self.encoder = seg_model.encoder
        self.decoder = seg_model.decoder
        self.segmentation_head = seg_model.segmentation_head

        # Get encoder output channels for classification head
        if hasattr(self.encoder, 'out_channels'):
            encoder_channels = self.encoder.out_channels[-1]
        else:
            # Default based on encoder name
            encoder_channels = 2048 if 'resnext50' in encoder_name else 512

        self.classification_head = ClassificationHead(encoder_channels, cls_classes)

    def forward(self, x, task='both'):
        encoder_features = self.encoder(x)

        outputs = {}

        if task in ['segmentation', 'both']:
            decoder_output = self.decoder(encoder_features)
            seg_output = self.segmentation_head(decoder_output)
            outputs['segmentation'] = seg_output

        if task in ['classification', 'both']:
            last_feature = encoder_features[-1]
            cls_output = self.classification_head(last_feature)
            outputs['classification'] = cls_output

        return outputs


def create_model(cfg):
    model = MultiTaskModel(
        encoder_name = getattr(cfg.model, 'encoder_name', 'resnext50_32x4d'),
        encoder_weights = getattr(cfg.model, 'encoder_weights', 'imagenet'),
        seg_classes = getattr(cfg.model, 'seg_classes', 3),
        cls_classes = getattr(cfg.model, 'cls_classes', 5),
        segmentation_head = getattr(cfg.model, 'segmentation_head', 'Segformer')
    )
    return model
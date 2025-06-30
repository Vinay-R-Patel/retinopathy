import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm


class ClassificationHead(nn.Module):
    def __init__(self, encoder_channels, num_classes=5, dropout_rate=0.5):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(encoder_channels, num_classes)

    def forward(self, x):
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class RetinoClassifier(nn.Module):
    def __init__(self, encoder_name="resnet34", num_classes=5):
        super().__init__()
        self.segmentation_model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )

        encoder_channels = self.segmentation_model.encoder.out_channels[-1]
        self.classification_head = ClassificationHead(encoder_channels, num_classes)

    def forward(self, x):
        encoder_features = self.segmentation_model.encoder(x)
        last_feature = encoder_features[-1]
        logits = self.classification_head(last_feature)
        return logits


class EfficientNetClassifier(nn.Module):
    def __init__(self, model_name="efficientnet_b2", num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class ResNetClassifier(nn.Module):
    def __init__(self, model_name="resnet34", num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class HRNetClassifier(nn.Module):
    def __init__(self, model_name="hrnet_w32", num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def create_model(cfg):
    model_config = cfg.model
    model_name = model_config.name
    num_classes = model_config.num_classes
    pretrained = getattr(model_config, 'pretrained', True)

    if "efficientnet" in model_name:
        model = EfficientNetClassifier(model_name, num_classes, pretrained=pretrained)
    elif "resnet" in model_name:
        model = ResNetClassifier(model_name, num_classes, pretrained=pretrained)
    elif "hrnet" in model_name:
        model = HRNetClassifier(model_name, num_classes, pretrained=pretrained)
    else:
        model = RetinoClassifier(model_name, num_classes)

    return model
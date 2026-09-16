import torch
import torch.nn as nn

# Optional import, only used for legacy/transfer-learning models.
try:
    import torchvision.models as models
except ImportError:
    models = None


def weights_init_he(m):
    """
    He/Kaiming initialization for all trainable layers.
    """

    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm3d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.apply(weights_init_he)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LandmarkCNN(nn.Module):
    def __init__(self, num_classes=7, dropout=0.5, num_landmarks=7):
        super(LandmarkCNN, self).__init__()
        self.num_landmarks = num_landmarks
        self.expand = nn.Conv2d(3, 64, kernel_size=1)
        self.expand_bn = nn.BatchNorm2d(64)
        self.features = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(num_landmarks * 256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        self.apply(weights_init_he)

    def forward(self, x):
        original_shape = x.shape
        if len(original_shape) == 5:
            B, N, C, H, W = original_shape
            x = x.view(B * N, C, H, W)
        elif len(original_shape) == 4:
            N, C, H, W = original_shape
            B = 1
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected input shape: {original_shape}")
        x = self.expand(x)
        x = self.expand_bn(x)
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(B, N * 256)
        x = self.classifier(x)
        if len(original_shape) == 4:
            x = x.squeeze(0)
        return x


class LandmarkResNet18(nn.Module):
    def __init__(
        self,
        num_classes=7,
        pretrained=False,
        freeze_backbone=False,
        dropout=0.5,
        num_landmarks=7
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        if pretrained:
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )
        else:
            self.backbone = models.resnet18(weights=None)
            self.backbone.apply(weights_init_he)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * num_landmarks, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        self.classifier.apply(weights_init_he)

    def forward(self, x):
        original_shape = x.shape
        if len(original_shape) == 5:
            B, N, C, H, W = original_shape
            x = x.view(B * N, C, H, W)
        elif len(original_shape) == 4:
            N, C, H, W = original_shape
            B = 1
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected input shape: {original_shape}")
        x = self.backbone(x)
        x = x.view(B, N * x.size(1))
        x = self.classifier(x)
        if len(original_shape) == 4:
            x = x.squeeze(0)
        return x


def setup_transfer_learning_model(
    model_name,
    num_classes,
    device,
    freeze_features=True
):
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.apply(weights_init_he)
    else:
        raise ValueError("Unsupported model")
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True
    return model.to(device)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

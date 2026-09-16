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


class S3D_CNN(nn.Module):
    """
    Model 3 : Shared 3D CNN
    Input: (B, N, 3, H, W)
    Output: (B, num_classes)
    """

    def __init__(self, num_landmarks=7, num_classes=7, dropout=0.4):
        super().__init__()

        self.num_landmarks = num_landmarks

        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            # Block 2
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            # Block 3
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            # Block 4
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.apply(weights_init_he)

    def forward(self, x):
        B, N, C, H, W = x.shape

        if N != self.num_landmarks:
            raise ValueError(
                f"Expected {self.num_landmarks} landmarks, "
                f"but received {N}."
            )

        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

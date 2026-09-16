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


class SCNN_FMF(nn.Module):
    """
    Model 2 : Shared CNN with Feature Map Fusion
    Input: (B, N, 3, H, W)
    Output: (B, num_classes)
    """

    def __init__(self, num_landmarks=7, num_classes=7, dropout=0.4):
        super().__init__()

        self.num_landmarks = num_landmarks

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(num_landmarks * 256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

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

        x = x.view(B * N, C, H, W)
        x = self.features(x)

        x = x.view(B, N, x.size(1), x.size(2), x.size(3))
        x = x.reshape(B, N * x.size(2), x.size(3), x.size(4))

        x = self.fusion(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

# models/__init__.py

from .standard_cnn import StandardCNN
from .scnn_fc import SCNN_FC
from .ecf_cnn import ECF_CNN
from .scnn_fmf import SCNN_FMF
from .s3d_cnn import S3D_CNN
from .lscnn_fc import LSCNN_FC
from .lscnn_fmf import LSCNN_FMF

# Legacy / additional models found in the supplied source.
from .legacy_models import (
    SimpleCNN,
    LandmarkCNN,
    LandmarkResNet18,
    setup_transfer_learning_model,
    count_trainable_parameters,
)

from .standard_cnn import weights_init_he

# cropping.py
"""
Patch extraction and patch transformation utilities.

The crop and transform logic is preserved from the original
landmark extraction implementation.
"""

from PIL import Image
from torchvision import transforms

# =====================================================
# Patch Transform
# =====================================================
normalize_patch = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

transform_patch = transforms.Compose([
    transforms.ToTensor(),
    normalize_patch
])


# =====================================================
# Patch Extraction
# =====================================================
def crop_patch(image, center_x, center_y, patch_size=48):
    img_width, img_height = image.size
    left = max(0, int(center_x - patch_size / 2))
    top = max(0, int(center_y - patch_size / 2))
    right = min(img_width, int(center_x + patch_size / 2))
    bottom = min(img_height, int(center_y + patch_size / 2))
    cropped = image.crop((left, top, right, bottom))
    cropped = cropped.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
    return cropped

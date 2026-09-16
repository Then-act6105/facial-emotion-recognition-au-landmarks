# landmarks.py
"""
MediaPipe Face Landmarker initialization, landmark detection, and
landmark-region feature extraction.

The extraction logic is preserved from the original implementation.
Region/AU definitions live in au_mapping.py and crop/transform logic
lives in cropping.py.
"""

import numpy as np
import torch
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from .au_mapping import (
    LANDMARK_REGIONS_7,
    AU_BASE_MAPPINGS,
    build_action_units,
)
from .cropping import crop_patch, transform_patch


# =====================================================
# MediaPipe Face Landmarker Initialization
# =====================================================
MODEL_PATH = "face_landmarker.task"  # Default path, update as needed

try:
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    print("MediaPipe initialized successfully")
except Exception as e:
    print(f"Warning: MediaPipe initialization failed: {e}. Landmarks will be zeros.")
    landmarker = None


# =====================================================
# Landmark Detection
# =====================================================
def detect_landmarks(image: Image.Image):
    if landmarker is None:
        return None
    image_np = np.array(image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    detection_result = landmarker.detect(mp_image)
    if not detection_result.face_landmarks:
        return None
    landmarks = detection_result.face_landmarks[0]
    landmark_coords = []
    img_w, img_h = image.size
    for landmark in landmarks:
        px = min(int(landmark.x * img_w), img_w - 1)
        py = min(int(landmark.y * img_h), img_h - 1)
        landmark_coords.append((px, py))
    return landmark_coords


# =====================================================
# Main Landmark Feature Extraction
# =====================================================
def extract_raw_landmark_features(
    image: Image.Image,
    patch_size=48,
    num_landmarks=7,
    with_au=False,
    au_list=None
):
    """
    Extract landmark patches from an image.

    Args:
        image: Input PIL Image
        patch_size: Size of each extracted patch
        num_landmarks: Number of patches to extract
        with_au: Whether to use AU-based regions instead of standard 7 regions
        au_list: Optional list of specific AUs to use (e.g., ["AU1", "AU2"]), uses all AUs if None

    Returns:
        torch.Tensor: Shape (num_landmarks, 3, patch_size, patch_size)
    """
    # Determine which regions to use
    if with_au:
        if au_list is not None:
            regions_dict = build_action_units(au_list)
        else:
            regions_dict = build_action_units()
        regions = list(regions_dict.values())
    else:
        regions_dict = LANDMARK_REGIONS_7
        regions = [{"landmark_indices": v} for k, v in regions_dict.items()]

    # Detect landmarks
    landmarks = detect_landmarks(image)
    patches = []

    if landmarks is None:
        # Return zero patches if no landmarks detected
        return torch.zeros((num_landmarks, 3, patch_size, patch_size), dtype=torch.float32)

    # Process regions up to num_landmarks
    for i in range(min(num_landmarks, len(regions))):
        region = regions[i]
        indices = region.get("landmark_indices", [])

        xs = []
        ys = []
        for idx in indices:
            if idx < len(landmarks):
                xs.append(landmarks[idx][0])
                ys.append(landmarks[idx][1])

        if len(xs) == 0:
            # Use image center if no valid landmark indices
            center_x = image.width / 2
            center_y = image.height / 2
        else:
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)

        patch = crop_patch(image, center_x, center_y, patch_size)
        patch = transform_patch(patch)
        patches.append(patch)

    # Pad with zero patches if we need more landmarks than regions
    while len(patches) < num_landmarks:
        patches.append(torch.zeros((3, patch_size, patch_size), dtype=torch.float32))

    # Trim if we have more patches than requested (unlikely, but just in case)
    patches = patches[:num_landmarks]

    return torch.stack(patches)


# Helper to get available regions
def get_available_regions(with_au=False):
    if with_au:
        return list(AU_BASE_MAPPINGS.keys())
    else:
        return list(LANDMARK_REGIONS_7.keys())

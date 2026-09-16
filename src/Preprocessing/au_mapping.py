# au_mapping.py
"""
Action Unit and MediaPipe landmark-region definitions.

This module contains the complete AU/region mapping logic from the
original landmark extraction implementation.
"""

# =====================================================
# Landmark Region Definitions
# =====================================================

# Standard 7 regions (default)
LANDMARK_REGIONS_7 = {
    "left_eye": [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173],
    "right_eye": [263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398],
    "left_eyebrow": [55, 65, 66, 105, 107],
    "right_eyebrow": [285, 295, 296, 334, 336],
    "nose": [1, 2, 4, 5, 6, 98, 99, 100, 101, 102, 103, 104, 196, 327, 328, 329, 330, 331, 333, 426, 427, 428, 429, 430, 431],
    "mouth": [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 95, 146, 178, 181, 185, 191, 308, 310, 311, 312, 314, 317, 318, 324, 375, 402, 405, 409, 415],
    "face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54]
}

# Dynamic Action Units (AUs) Base Definitions
AU_BASE_MAPPINGS = {
    "AU1": {
        "name": "Inner Brow Raiser",
        "regions": ["left_eyebrow_inner", "right_eyebrow_inner", "nose_root"]
    },
    "AU2": {
        "name": "Outer Brow Raiser",
        "regions": ["left_eyebrow_outer", "right_eyebrow_outer"]
    },
    "AU4": {
        "name": "Brow Lower",
        "regions": ["left_eyebrow", "right_eyebrow", "nose_root"]
    },
    "AU5": {
        "name": "Upper Lid Raiser",
        "regions": ["left_eye", "right_eye"]
    },
    "AU6": {
        "name": "Cheek Raiser",
        "regions": ["left_cheek", "right_cheek"]
    },
    "AU7": {
        "name": "Lid Tightener",
        "regions": ["left_eye", "right_eye"]
    },
    "AU9": {
        "name": "Nose Wrinkler",
        "regions": ["nose", "upper_lip"]
    },
    "AU10": {
        "name": "Upper Lip Raiser",
        "regions": ["nose", "upper_lip"]
    },
    "AU12": {
        "name": "Lip Corner Puller",
        "regions": ["left_mouth_corner", "right_mouth_corner"]
    },
    "AU15": {
        "name": "Lip Corner Depressor",
        "regions": ["left_mouth_corner", "right_mouth_corner"]
    },
    "AU16": {
        "name": "Lower Lip Depressor",
        "regions": ["lower_lip", "chin_upper"]
    },
    "AU17": {
        "name": "Chin Raiser",
        "regions": ["chin", "lower_lip"]
    },
    "AU20": {
        "name": "Lip Stretcher",
        "regions": ["left_mouth_corner", "right_mouth_corner"]
    },
    "AU23": {
        "name": "Lip Tightener",
        "regions": ["upper_lip", "lower_lip"]
    },
    "AU24": {
        "name": "Lip Pressor",
        "regions": ["upper_lip", "lower_lip"]
    },
    "AU25": {
        "name": "Lips Part",
        "regions": ["upper_lip", "lower_lip"]
    },
    "AU26": {
        "name": "Jaw Drop",
        "regions": ["jawline", "chin"]
    }
}

# MediaPipe 468 Landmark Topology - Region Definitions
MEDIAPIPE_REGION_LANDMARKS = {
    "nose_root": [1, 2, 4, 5],
    "nose": [1, 2, 4, 5, 6, 98, 99, 100, 101, 102, 103, 104, 196, 327, 328, 329, 330, 331, 333, 426, 427, 428, 429, 430, 431],
    "left_eyebrow": [55, 65, 66, 105, 107],
    "left_eyebrow_inner": [55, 65],
    "left_eyebrow_outer": [105, 107],
    "right_eyebrow": [285, 295, 296, 334, 336],
    "right_eyebrow_inner": [285, 295],
    "right_eyebrow_outer": [334, 336],
    "left_eye": [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173],
    "right_eye": [263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398],
    "left_cheek": [33, 133, 28, 38, 42],
    "right_cheek": [263, 362, 258, 268, 272],
    "upper_lip": [13, 14, 17],
    "lower_lip": [146, 178, 181, 185, 191],
    "left_mouth_corner": [61, 78, 80, 81, 82, 84, 87, 88, 95],
    "right_mouth_corner": [308, 310, 311, 312, 314, 317, 318],
    "mouth": [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 95, 146, 178, 181, 185, 191, 308, 310, 311, 312, 314, 317, 318, 324, 375, 402, 405, 409, 415],
    "chin_upper": [146, 178],
    "chin": [152, 146, 178, 181, 185, 191],
    "jawline": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54],
    "face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54]
}


def get_au_landmarks(au_codes):
    """
    Dynamically resolve landmark indices for user-provided list of AUs at runtime.
    Combines all relevant MediaPipe landmarks from regions associated with each input AU.
    """
    all_landmarks = set()
    for au in au_codes:
        if au in AU_BASE_MAPPINGS:
            au_regions = AU_BASE_MAPPINGS[au]["regions"]
            for region in au_regions:
                if region in MEDIAPIPE_REGION_LANDMARKS:
                    all_landmarks.update(MEDIAPIPE_REGION_LANDMARKS[region])
    return sorted(list(all_landmarks))


def build_action_units(au_list=None):
    """
    Runtime builder to generate ACTION_UNITS dictionary with resolved landmark indices.
    Uses user-provided AU list to dynamically load only required AUs.
    """
    target_aus = au_list if au_list is not None else list(AU_BASE_MAPPINGS.keys())
    action_units = {}
    for au in target_aus:
        if au in AU_BASE_MAPPINGS:
            action_units[au] = {
                "name": AU_BASE_MAPPINGS[au]["name"],
                "landmark_indices": get_au_landmarks([au])
            }
    return action_units


def build_au_region_plan(au_list):
    """
    Build a plan for AU-based landmark regions.

    Args:
        au_list (list): List of AU codes to use

    Returns:
        dict: Plan with "landmark_groups" key containing the AU regions
    """
    # Validate AU codes
    valid_aus = []
    invalid_aus = []
    for au_code in au_list:
        if au_code in AU_BASE_MAPPINGS:
            valid_aus.append(au_code)
        else:
            invalid_aus.append(au_code)

    if invalid_aus:
        raise ValueError(f"Invalid AU codes provided: {invalid_aus}. Valid codes are: {list(AU_BASE_MAPPINGS.keys())}")

    if not valid_aus:
        raise ValueError("No valid AU codes provided.")

    # Build landmark groups
    landmark_groups = []
    for au_code in valid_aus:
        au_info = AU_BASE_MAPPINGS[au_code]
        landmark_groups.append({
            "name": f"{au_code}_{au_info['name']}",
            "au_code": au_code,
            "regions": au_info["regions"],
            "landmark_indices": get_au_landmarks([au_code])
        })

    return {
        "landmark_groups": landmark_groups,
        "au_list": valid_aus
    }

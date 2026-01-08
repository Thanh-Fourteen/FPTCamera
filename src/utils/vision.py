import numpy as np
from typing import List

def crop_image(image: np.ndarray, box: List[int]) -> np.ndarray:
    """
    Crop a region from the image based on the bounding box.

    Args:
        image (np.ndarray): The input image (H, W, C).
        box (List[int]): Bounding box in [x1, y1, x2, y2] format.

    Returns:
        np.ndarray: Cropped image region.
    """
    if len(box) != 4:
        raise ValueError("Bounding box must contain exactly 4 coordinates.")

    x1, y1, x2, y2 = box

    # Ensure coordinates are within the image boundaries
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # Ensure valid cropping area
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop coordinates: {box}")

    return image[y1:y2, x1:x2]
import cv2
import numpy as np
from typing import Optional, Dict
from shapely.geometry import Polygon

def visualize_polygon(image: np.ndarray, class_counts: Dict[str, int], polygon: Optional[Polygon]) -> np.ndarray:
    """
    Visualize the polygon on the image.

    Args:
        image (np.ndarray): The image to visualize the polygon on.
        polygon (Polygon): The polygon to visualize.

    Returns:
        np.ndarray: The image with the polygon drawn on it.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array.")

    if isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy
        points = np.array(list(zip(x, y)), dtype=np.int32)
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    for idx, (class_name, count) in enumerate(class_counts.items()):
        cv2.putText(
            image,
            f"Count: {count} {class_name}",
            (10, 30 + idx * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    return image
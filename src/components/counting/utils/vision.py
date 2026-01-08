import cv2
import numpy as np
from typing import Optional, Dict
from shapely.geometry import Polygon


def visualize_polygon(
    image: np.ndarray,
    class_counts: Dict[str, int],
    polygon: Optional[Polygon],
    polygon_color=(0, 255, 0),
    fill_alpha=0.3,
) -> np.ndarray:
    """
    Visualize polygon and class counts nicely on image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array.")

    output = image.copy()
    overlay = image.copy()

    # ===== Draw polygon =====
    if isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy
        points = np.array(list(zip(x, y)), dtype=np.int32)

        # Fill polygon (transparent)
        cv2.fillPoly(overlay, [points], polygon_color)

        # Draw border (anti-aliased)
        cv2.polylines(
            output,
            [points],
            isClosed=True,
            color=polygon_color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )

        # Blend overlay
        output = cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0)

    # ===== Draw class counts with background =====
    x0, y0 = 20, 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    thickness = 2
    padding_x = 14
    padding_y = 10
    line_spacing = 18

    for idx, (class_name, count) in enumerate(class_counts.items()):
        text = f"{class_name}: {count}"

        (tw, th), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        y = y0 + idx * (th + line_spacing)

        # Background box (to hơn)
        cv2.rectangle(
            output,
            (x0 - padding_x, y - th - padding_y),
            (x0 + tw + padding_x, y + baseline + padding_y),
            (0, 0, 0),
            -1,
        )

        # Text
        cv2.putText(
            output,
            text,
            (x0, y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    return output

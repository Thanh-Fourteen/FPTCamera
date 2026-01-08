from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from collections import defaultdict

from shapely.geometry import Point, Polygon 

from utils.dtos import Detection, FrameData
from utils.logger import LoggerManager
from components.counting.base import BaseCounting
from components.counting.utils.vision import visualize_polygon

class PolygonDirectionalCounter(BaseCounting):
    """
    Count people entering or exiting a polygon area based on movement direction.
    - Uses shapely LineString to detect crossing polygon boundaries.
    - Compares movement vector angle with the configured 'in' direction vector.
    """
    MERGE_CLASSES = {
        "human": "person",
    }
    def __init__(self) -> None:
        """
        Initialize the ObjectCounter with default values.
        """
        self.polygon: Optional[Polygon] = None
        self.include_classes: Set[str] = set()
        self.exclude_classes: Set[str] = set()
        self.logger: LoggerManager
        self.visualize = False
        self.visualize_polygon = None

    def initialize(self, config: Dict[str, Any], logger: LoggerManager = None) -> None:
        """
        Set up the counting strategy with the given configuration and logger.

        Args:
            config (Dict[str, Any]): Configuration for the counting strategy.
            logger (LoggerManager): Logger instance for logging.
        """
        self.logger = logger

        # Check for coordinates to avoid KeyError
        if "coordinates" in config:
            coords = config["coordinates"]
            self._raw_coordinates = coords 
            self.polygon = None

        # Get include/exclude classes from config
        self.include_classes = set(config.get("include_classes", []))
        self.exclude_classes = set(config.get("exclude_classes", []))

        # Get visualization options
        self.visualize = config.get("visualize", False)

        # Cache the visualize function if visualization is enabled
        if self.visualize:
            self.visualize_polygon = visualize_polygon

    def _filter_detections(
        self, detections: Sequence[Detection]
    ) -> List[Detection]:
        """
        Filter detections based on include/exclude classes.

        Args:
            detections (Sequence[Detection]): Sequence of detections to filter.

        Returns:
            List[Detection]: Filtered list of detections.
        """
        if not self.include_classes and not self.exclude_classes:
            return list(detections)
        return [
            detection
            for detection in detections
            if (
                not self.include_classes or detection.label in self.include_classes
            )
            and detection.label not in self.exclude_classes
        ]

    def execute(self, data: FrameData) -> Dict[str, int]:
        """
        Execute the crowd counting strategy on the provided image data.

        Args:
            data (FrameData): The frame data containing detections.

        Returns:
            Dict[str, int]: A dictionary mapping class names to their counts.
        """
        if getattr(self, "_raw_coordinates", None) is not None:
            coords = getattr(self, "_raw_coordinates")
            if isinstance(coords, list) and len(coords) > 0:
                h, w = data.image.shape[:2]
                is_normalized = all(isinstance(c, (list, tuple)) and 0.0 <= c[0] <= 1.0 and 0.0 <= c[1] <= 1.0 for c in coords)
                if is_normalized:
                    abs_coords = [(c[0] * w, c[1] * h) for c in coords]
                else:
                    abs_coords = coords
                self.polygon = Polygon(abs_coords)
            # Clear raw coordinates after building
            delattr(self, "_raw_coordinates")

        # Filter detections first
        filtered_detections = self._filter_detections(data.detection)

        # Initialize count dictionary
        class_counts = defaultdict(int)

        # If no polygon defined, count all filtered detections
        if self.polygon is None:
            for detection in filtered_detections:
                class_counts[
                        self.MERGE_CLASSES.get(detection.label, detection.label)
                    ] += 1
        else:
            # Count objects inside polygon by class
            for detection in filtered_detections:
                if self.is_point_inside_polygon(
                    ((detection.box.x1 + detection.box.x2) / 2, detection.box.y2)
                ):
                    class_counts[
                        self.MERGE_CLASSES.get(detection.label, detection.label)
                    ] += 1

        # Visualize
        if self.visualize and self.visualize_polygon:
            data.image = self.visualize_polygon(data.image, dict(class_counts),self.polygon)
        return dict(class_counts)

    def is_point_inside_polygon(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside the polygon.

        Args:
            point (Tuple[float, float]): (x, y) coordinates of the point to check.

        Returns:
            bool: True if point is inside the polygon, False otherwise.
        """
        return self.polygon is not None and self.polygon.contains(Point(point))

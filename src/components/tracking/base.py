from abc import abstractmethod

from utils.dtos import TrackingObject, FrameData
from components.base_strategy import IStrategy


class BaseTracking(IStrategy):
    """Interface for tracking strategy classes.

    Any tracking strategy must implement `execute`, which receives FrameData
    (with detections) and returns TrackingResults.
    """

    @abstractmethod
    def execute(self, data: FrameData) -> list[TrackingObject]:
        """Associates detections across frames to maintain object identities.

        Args:
            data (FrameData): Contains image, detections, and metadata.

        Returns:
            TrackingResults: An object containing the aggregated tracking results
                            (tracked objects with IDs and bounding boxes) for the current frame.
        """
        pass

from abc import abstractmethod

from components.base_strategy import IStrategy
from utils.dtos import Detection, FrameData


class BaseDetection(IStrategy):
    """
    Interface (Abstract Base Class) for object detection strategies.

    All concrete detection strategies must implement the `execute` method.
    This interface defines the contract for how a detection algorithm
    should operate, promoting flexibility and extensibility.
    """

    @abstractmethod
    def execute(self, data: FrameData) -> list[Detection]:
        """
        Performs object detection on the given image frame.

        Args:
            frame (FrameData): A list of input requests, each containing image data
                                    and associated metadata (camera_id, frame_id).

        Returns:
            List[Detection]: An object containing the frame ID and a list of detected objects.
                              This includes their bounding boxes, scores, and labels.
        """
        pass

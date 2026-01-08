import cv2
from typing import Any, Dict

from utils.dtos import FrameData
from core.base_module import BaseAIModule
from components.counting.base import BaseCounting

class CrowdCounter(BaseAIModule):
    """
    CrowdCounter module for counting detected objects.

    This module takes frame data, applies a counting strategy, and updates
    the frame data with the crowd count. It supports visualization of the
    count on the frame.

    Attributes:
        strategy (BaseCounting): The counting strategy to be used.
    """

    def __init__(self, strategy: BaseCounting) -> None:
        """Initializes the CrowdCounter module.

        Args:
            strategy (BaseCounting): The counting strategy to use.

        Raises:
            TypeError: If the provided strategy is not an instance of BaseCounting.
        """
        super().__init__()
        self._module_name = "CrowdCounter"
        if not isinstance(strategy, BaseCounting):
            raise TypeError("Strategy must be an instance of BaseCounting")
        self.strategy = strategy

    def process(self, frame_data: FrameData) -> FrameData:
        """Processes a single frame to count objects.

        Args:
            frame_data (FrameData): The input frame data containing the image and
                                    other relevant information.

        Returns:
            FrameData: The updated frame data with the crowd count.
        """
        self.logger.log_debug(
            self.camera_id,
            self.area_name,
            f"Processing frame ID: {frame_data.frame_id}",
        )
        try:
            counting_results = self.strategy.execute(frame_data)
            frame_data.crowd_count = sum(counting_results.values())

        except Exception as e:
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Error processing frame ID {frame_data.frame_id}: {e}",
                exc_info=True,
            )
            frame_data.crowd_count = 0
        return frame_data

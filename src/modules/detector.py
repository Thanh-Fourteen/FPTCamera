from core.base_module import BaseAIModule
from components.detection import BaseDetection
from utils.dtos import FrameData


class Detector(BaseAIModule):
    """AI module that handles object detection using a given detection strategy.

    Attributes:
        strategy (BaseDetection): Concrete detection algorithm implementing
                                  `execute(FrameData) -> List[Detection]`.
    """

    def __init__(self, strategy: BaseDetection):
        """Initialize the Detector module with a specific detection strategy.

        Args:
            strategy (BaseDetection): An instance of a detection strategy.

        Raises:
            TypeError: If the strategy is not an instance of BaseDetection.
        """
        super().__init__()
        self._module_name = "Detector"

        if not isinstance(strategy, BaseDetection):
            raise TypeError("Strategy must be an instance of BaseDetection")
        self.strategy = strategy

    def process(self, frame_data: FrameData) -> FrameData:
        """Apply the detection strategy on the input frame and attach results.

        Args:
            frame_data (FrameData): Frame data including the image and metadata.

        Returns:
            FrameData: Updated frame data with detection results.
        """
        try:
            # Execute the detection strategy on the image data
            detection_results = self.strategy.execute(frame_data)

            if frame_data.detection is None:
                frame_data.detection = detection_results
            else:
                frame_data.detection.extend(detection_results)

            self.logger.log_debug(
                self.camera_id,
                self.area_name,
                f"Detected {len(detection_results)} objects "
                f"in frame ID: {frame_data.frame_id}",
            )

        except Exception as e:
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Error processing frame ID {frame_data.frame_id}: {e}",
                exc_info=True,
            )
            frame_data.detection = []

        return frame_data
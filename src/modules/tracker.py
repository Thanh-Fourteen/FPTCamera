from core.base_module import BaseAIModule
from components.tracking.base import BaseTracking
from utils.dtos import FrameData


class Tracker(BaseAIModule):
    def __init__(self, strategy: BaseTracking):
        super().__init__()
        self._module_name = "Tracker"

        if not isinstance(strategy, BaseTracking):
            raise TypeError("Strategy must be an instance of BaseTracking")
        self.strategy = strategy

    def process(self, frame_data: FrameData) -> FrameData:
        try:
            if frame_data.image is None:
                self.logger.log_warning(
                    self.camera_id,
                    self.area_name,
                    f"Frame ID {frame_data.frame_id} has no image data.",
                )
            if not frame_data.detection or not frame_data.detection:
                self.logger.log_warning(
                    self.camera_id,
                    self.area_name,
                    f"Frame ID {frame_data.frame_id} has no detection data.",
                )

            # Execute the detection strategy on the image data
            tracking_results = self.strategy.execute(frame_data)

            # Add results to the frame data
            frame_data.track = tracking_results

        except Exception as e:
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Error processing Tracker frame ID {frame_data.frame_id}: {e}",
                exc_info=True,
            )

        return frame_data

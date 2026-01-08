import numpy as np
from typing import Optional
from trism_cv import TritonModel

from utils.logger import LoggerManager
from utils.dtos import Detection, BoundingBox, FrameData
from components.detection.base import BaseDetection


class HumanDetection(BaseDetection):
    """Concrete implementation of BaseDetection using trism-cv and Triton Inference Server."""

    def __init__(self):
        self.model: Optional[TritonModel] = None
        self.logger: LoggerManager

    def initialize(self, config, logger) -> None:
        """Initialize Triton model using trism-cv.

        Args:
            config (dict): A dictionary containing configuration parameters for the Triton model
            logger (LoggerManager): An instance of LoggerManager for logging.
        """
        self.logger = logger
        self.camera_id = config.get("camera_id", "unknown")
        self.area_name = config.get("area_name", "unknown_area")
        self.threshold = config.get("threshold", 0.5)

        self.model_name = config.get("model_name", "")
        self.version = config.get("version", 0)
        self.url = config.get("url", "")
        self.use_grpc = config.get("use_grpc", True)
        self.id2label = config.get("class_names", {})
        self.id2label = {int(k): v for k, v in self.id2label.items()}

        try:
            self.model = TritonModel(
                model=self.model_name,
                version=self.version,
                url=self.url,
                grpc=self.use_grpc,
            )
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.run([dummy_input])
            self.logger.log_info(
                self.camera_id,
                self.area_name,
                f"Initialize Triton Human Detection success",
            )
        except Exception as e:
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Error initializing Triton Human Detection: {e}",
                exc_info=True,
            )

    def execute(self, data: FrameData) -> list[Detection]:
        """Execute object detection for a list of requests using trism-cv and Triton.

        Args:
            frame (FrameData): Input request containing image data
                                    and associated metadata (camera_id, frame_id).

        Returns:
            List[Detection]: Detection results for all images across all requests.

        Raises:
            ValueError: If no valid images are provided in any request.
        """
        if not self.model:
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                "Triton Human Detection is not initialized. Cannot perform detection.",
            )
            return []

        if data is None or data.image is None:
            raise ValueError("FrameData is empty or has no image.")

        outputs = self.model.run(data={"INPUT": [data.image]})
        if not outputs:
            return []

        detections_list = self._postprocess_detections(outputs[0], self.threshold)

        return detections_list

    def _postprocess_detections(self, raw_output, threshold) -> list[Detection]:
        """Convert raw Triton output to list of Detection objects.

        Args:
            raw_output: Raw output from Triton model (expected as numpy array).

        Returns:
            List[Detection]: List of Detection objects.
        """
        detections = []
        for det in raw_output:
            if det[4] < threshold:
                continue
            try:
                detection = Detection(
                    box=BoundingBox.fromlist(det[:4]),
                    confidence=det[4],
                    label=self.id2label.get(int(det[5]), "unknown"),
                )
                detections.append(detection)

            except Exception as e:
                self.logger.log_warning(
                    self.camera_id,
                    self.area_name,
                    f"Skipping invalid detection entry: {det.tolist()[:6]} ({e})",
                )

        return detections

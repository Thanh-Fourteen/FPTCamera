"""ByteTrack strategy implementation.

Implements ITrackingStrategy using the BYTETracker and STrack classes
to associate detections over time.
"""

import numpy as np
from typing import Any, Dict, List

from components.tracking.base import BaseTracking
from components.tracking.ByteTrack.byte_tracker import STrack, BYTETracker
from components.tracking.ByteTrack.byte_tracker import BYTETracker
from utils.dtos import BoundingBox, FrameData, TrackingObject
from utils.logger import LoggerManager


class ByteTrack(BaseTracking):
    """
    ByteTrack implementation for tracking objects in video frames.
    """

    def __init__(self) -> None:
        """
        Initialize the ByteTrack tracker.
        """
        self.tracker: BYTETracker
        self.logger: LoggerManager

    def initialize(self, config: Dict[str, Any], logger: LoggerManager) -> None:
        """
        Setup the ByteTrack tracker with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing parameters for the tracker.
            logger (Logger): Logger instance for logging messages.
        """
        self.logger = logger
        self.camera_id = config.get("camera_id", "unknown")
        self.area_name = config.get("area_name", "unknown_area")
        track_thresh = config.get("track_thresh", 0.5)
        match_thresh = config.get("match_thresh", 0.5)
        track_buffer = config.get("track_buffer", 30)
        use_iou_score_only = config.get("use_iou_score_only", False)

        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            use_iou_score_only=use_iou_score_only,
        )
        self.logger.log_info(
            self.camera_id,
            self.area_name,
            f"ByteTrack tracker initialized with configuration: {config}",
        )

    def _build_tracking_objects(
        self, stracks_list: List[STrack], original_indices: List[int], data: FrameData
    ) -> List[TrackingObject]:
        """Build tracking objects from tracker results."""
        tracking_objects = []
        for current_strack, original_det_idx in zip(stracks_list, original_indices):
            if not (0 <= original_det_idx < len(data.detection)):
                self.logger.log_warning(
                    self.camera_id,
                    self.area_name,
                    f"STrack {current_strack.track_id} has invalid original_det_idx {original_det_idx}. Skipping.",
                )
                continue

            original_detection = data.detection[original_det_idx]
            tracking_objects.append(
                TrackingObject(
                    track_id=current_strack.track_id,
                    prev_state=current_strack.prev_state,
                    box=BoundingBox.fromlist(current_strack.tlbr),
                    label=original_detection.label,
                    confidence=current_strack.score,
                )
            )
            current_strack.prev_state = original_detection.state
        return tracking_objects

    def execute(self, data: FrameData) -> List[TrackingObject]:
        """
        Execute the ByteTrack algorithm on the given frame data.

        Args:
            data (FrameData): Frame data containing detection results and other information.

        Returns:
            List[TrackingResult]: List of tracking results after processing the frame data.
        """
        self.logger.log_debug(
            self.camera_id,
            self.area_name,
            f"Executing ByteTrack on frame: {data.frame_id}",
        )

        # Check if the tracker has been initialized with a frame rate
        if self.tracker.frame_rate == -1:
            self.tracker.change_frame_rate(data.metadata.get("frame_rate", 30))
            self.logger.log_debug(
                self.camera_id,
                self.area_name,
                f"Updated tracker frame rate to: {self.tracker.frame_rate} fps",
            )

        # Combine extraction of boxes and scores in one pass
        detection_info = [[*det.box.tolist(), det.confidence] for det in data.detection]
        detections_array = np.asarray(detection_info)
        stracks_list, original_indices = self.tracker.update(detections_array)

        tracking_objects = self._build_tracking_objects(
            stracks_list, original_indices, data
        )

        # final_tracking_results = TrackingResult(
        #     camera_id=data.camera_id,
        #     frame_id=data.frame_id,
        #     objects=tracking_objects
        # )
        self.logger.log_debug(
            self.camera_id,
            self.area_name,
            f"Tracking results for frame {data.frame_id}: {str(tracking_objects)}",
        )
        return tracking_objects

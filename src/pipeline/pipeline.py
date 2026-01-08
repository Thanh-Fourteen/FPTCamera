import os
import cv2
import numpy as np
from typing import Optional, Any, List, cast
from pathlib import Path
from datetime import datetime

from input import (
    InputSource,
    VideoInputSource,
    ImageInputSource,
    RTSPInputSource,
    IMG_FORMATS,
    FrameProducer,
)
from modules.detector import Detector
from core.base_module import IAIModule
from utils.dtos import FrameData
from utils.config import ConfigManager
from utils.logger import LoggerManager
from utils.plot import get_colors, visualize


class Pipeline:
    def __init__(
        self,
        config: ConfigManager,
        logger: LoggerManager,
        modules: List[IAIModule],
    ):
        self.config = config
        self.logger = logger
        self.modules = modules

        # Load application configuration
        app_config = self.config.get("application", {})
        self.camera_id = app_config.get("camera_id", "unknow_camera")
        self.area_name = app_config.get("area_name", "unknow_area")
        self.timestamp = app_config.get("timestamp", None)
        self.visualize_type = app_config.get("visualize", None)

        # Load input config
        input_config = app_config.get("input", {})
        self.rotation_degrees = input_config.get("rotate_degrees", 0)
        self.target_fps = input_config.get("target_fps", 10)
        self.skip_frames = input_config.get("skip_frames", -1)
        self.buffer_strategy = input_config.get("buffer_strategy", "fixed")
        self.buffer_size = input_config.get("buffer_size", 10)
        self.codec = input_config.get("codec", "XVID")

        # Load output config
        output_config = app_config.get("output", {})
        self.output_path = output_config.get("output_path", "")

        # Load Frame Producer
        producer_config = app_config.get("frame_producer", {})
        self.reconnect_attempts = producer_config.get("reconnect_attempts", 3)
        self.reconnect_timeout = producer_config.get("reconnect_timeout_s", 5)

        # Init state
        self.display = False
        self.visualize = True
        self._current_source: Optional[InputSource] = None
        self._frame_producer: Optional[FrameProducer] = None
        self._initial_source_fps = 0.0
        self.frame_rate = 0.0

        # Initialize visualization settings
        self._initialize_visualization_settings()

        self.logger.log_info(
            self.camera_id,
            self.area_name,
            f"Pipeline initialized with {len(self.modules)} modules.",
        )
        for module in self.modules:
            self.logger.log_info(
                self.camera_id,
                self.area_name,
                f" -> Enabled module: {module.name}",
            )

    def _initialize_visualization_settings(self) -> None:
        """Sets up visualization maps based on configured modules.

        Extracts class names from detector modules and prepares appropriate
        color maps for visualization.
        """
        # Load visualize names for visualization from modules
        visualize_map = {}
        self.vis_overlays = {}
        for module in self.modules:
            if hasattr(module.strategy, "id2label"):
                if (
                    (
                        self.visualize_type == "detection"
                        and isinstance(module, Detector)
                    )
                ):
                    strategy = cast(Any, module.strategy)
                    class_names = getattr(strategy, "id2label", {})
                    for class_name in class_names.values():
                        visualize_map[class_name] = len(visualize_map)

                    self.logger.log_debug(
                        self.camera_id,
                        self.area_name,
                        f"Added class names from {module.name}: {class_names}",
                    )

            if module.name not in ["Detector", "Tracker"]:
                self.vis_overlays[module.name] = module._config.get("visualize", None)

        self.logger.log_info(
            self.camera_id,
            self.area_name,
            f" ===> Enhance visualization settings for module: {self.vis_overlays}",
        )

        if self.visualize_type == "detection":
            cmap = get_colors(num_classes=len(visualize_map), filter_colors=True)
            self.cmap = {k: cmap[i] for i, k in enumerate(visualize_map.keys())}
        elif self.visualize_type == "tracking":
            colors = get_colors(
                num_classes=300, filter_colors=True
            )  # Default 300 colors for tracking
            self.cmap = {i: colors[i] for i, _ in enumerate(colors)}
        else:
            # Error handling for unsupported visualization types
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Unsupported visualization type: {self.visualize_type}. Defaulting to 'none'.",
            )
            self.visualize = False

    def run(self, media_source: Optional[Any] = None) -> None:
        """Runs the pipeline on a given media source or with a producer."""
        if not media_source:
            self.logger.log_error(
                self.camera_id, self.area_name, f"No media_source provided. Exiting."
            )
            return

        try:
            is_url = media_source.lower().startswith(
                ("rtsp://", "rtmp://", "http://", "https://")
            )
            if is_url:  # streaming
                self._current_source = RTSPInputSource(
                    rotation_degrees=self.rotation_degrees,
                    camera_id=self.camera_id,
                    area_name=self.area_name,
                    logger=self.logger,
                )
            else:
                file_ext = Path(media_source).suffix.lower()
                if file_ext in IMG_FORMATS:
                    self._current_source = ImageInputSource(
                        rotation_degrees=self.rotation_degrees,
                        camera_id=self.camera_id,
                        area_name=self.area_name,
                        logger=self.logger,
                    )
                else:
                    self._current_source = VideoInputSource(
                        rotation_degrees=self.rotation_degrees,
                        camera_id=self.camera_id,
                        area_name=self.area_name,
                        logger=self.logger,
                    )

            if not self._current_source.open(media_source):
                self.logger.log_error(
                    self.camera_id,
                    self.area_name,
                    f"Failed to open input source: {media_source}. Exiting.",
                )
                return

            if isinstance(self._current_source, ImageInputSource):
                success, img = self._current_source.read()
                if success:
                    fd = self._process_frame(img, 0)
                    self._display_output(fd)
                    if self.output_path:
                        cv2.imwrite(self.output_path, fd.image)
                    self._current_source.close()
            else:
                processed_frame_count = 0
                writer = None

                # Calculate FPS and skip settings from producer source
                self._initialize_fps_and_skip_settings(self._current_source.get_fps())

                self.logger.log_info(
                    self.camera_id, self.area_name, "Starting main processing loop..."
                )

                # last_processed_time = time.perf_counter()
                while True:
                    success, frame = self._current_source.read()

                    if not success:
                        if not self._current_source.is_opened():
                            self.logger.log_info(
                                self.camera_id,
                                self.area_name,
                                "Producer stopped and buffer depleted. Exiting run loop.",
                            )
                        break

                    if writer is None:
                        try:
                            writer = self._setup_video_writer(frame)
                            if writer is not None:
                                self.logger.log_info(
                                    self.camera_id,
                                    self.area_name,
                                    f"Video writer initialized to {self.output_path}",
                                )
                        except Exception as e:
                            self.logger.log_error(
                                self.camera_id,
                                self.area_name,
                                f"Failed to initialize video writer: {e}",
                            )

                    if processed_frame_count % (self._skip_interval) == 0:
                        fd = self._process_frame(frame, processed_frame_count)
                        if self.visualize and hasattr(fd, "image"):
                            if not self._display_output(fd):
                                self.logger.log_info(
                                    self.camera_id,
                                    self.area_name,
                                    "User requested exit.",
                                )
                                break
                            writer.write(fd.image)

                    processed_frame_count += 1

        except Exception as e:
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Pipeline run error: {e}",
                exc_info=True,
            )

        finally:
            self.logger.log_info(
                self.camera_id, self.area_name, "Pipeline run finished. Cleaning up."
            )

            if self._current_source and self._current_source.is_opened():
                self._current_source.close()

            if self.visualize:
                cv2.destroyAllWindows()
            self.logger.log_info(
                self.camera_id, self.area_name, "Pipeline cleanup complete."
            )

    def _initialize_fps_and_skip_settings(self, source_fps: float) -> None:
        skip = 0
        self._initial_source_fps = source_fps
        self.frame_rate = source_fps
        # Case 1: User-defined skip frames
        if self.skip_frames > 0:
            skip = self.skip_frames
            if self._initial_source_fps > 0 and (skip + 1) > 0:
                self.target_fps = self._initial_source_fps / (skip + 1)
        # Case 2: Auto-calculate based on target_fps
        else:
            if self._initial_source_fps > 0 and self.target_fps > 0:
                if self._initial_source_fps < self.target_fps:
                    skip = 0
                else:
                    skip = max(
                        0, int(round(self._initial_source_fps / self.target_fps) - 1)
                    )
            else:
                skip = 0
                if self._initial_source_fps == 0 and self.target_fps > 0:
                    self.logger.log_warning(
                        self.camera_id,
                        self.area_name,
                        "Source FPS is 0 or unavailable for auto-skip. Defaulting to no skip.",
                    )
        # Update target_fps based on calculation
        if self._initial_source_fps > 0:
            self.target_fps = self._initial_source_fps / (skip + 1)

        skip_interval = skip + 1
        if skip_interval <= 0:
            skip_interval = 1

        self._skip_interval = skip_interval

    async def process_frame(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        """Processes a single frame for streaming and returns the visualized frame."""

        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            self.logger.log_warning(
                self.camera_id,
                self.area_name,
                f"Received empty frame (ID: {frame_id}). Skipping.",
            )
            return frame

        try:
            frame_copy = frame.copy()
        except Exception as e:
            self.logger.log_warning(
                self.camera_id,
                self.area_name,
                f"Could not copy frame (ID: {frame_id}) using original: {e}",
            )
            frame_copy = frame

        try:
            timestamp = datetime.now()
            frame_data = FrameData(
                camera_id=self.camera_id,
                frame_id=frame_id,
                image=frame_copy,
                timestamp=timestamp,
            )
        except Exception as e:
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Error initializing FrameData for frame {frame_id}: {e}",
            )
            return frame_copy

        try:
            for module in self.modules:
                frame_data = module.process(frame_data)
        except Exception as e:
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Error processing frame {frame_id} with AI modules: {e}",
            )
            return frame_copy

        if self.visualize:
            items, statuses, cls_ids = [], [], []
            if self.visualize_type == "detection":
                items = frame_data.detection
                statuses = [
                    f"{getattr(obj, 'label', 'unknown')} {getattr(obj, 'confidence', 0):.2f}"
                    for obj in items
                ]
                cls_ids = [getattr(obj, "label", "unknown") for obj in items]

            elif self.visualize_type == "tracking":
                for obj in frame_data.track:
                    if getattr(obj, "label", "Unknown").lower() not in (
                        "body",
                        "human",
                    ):
                        continue

                    track_id = getattr(obj, "track_id", "Unknown")
                    status_text = f"ID-{track_id}"
                    cls_id = track_id

                    items.append(obj)
                    statuses.append(status_text)
                    cls_ids.append(cls_id)

            bboxes = [d.box.tolist() for d in items]

            frame_data.image = visualize(
                image=frame_data.image,
                bboxes=bboxes,
                class_ids=cls_ids,
                statuses=statuses,
                cmap=self.cmap,
            )

        return frame_data.image

    def _process_frame(self, frame: np.ndarray, frame_id: int) -> Optional[FrameData]:
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            self.logger.log_warning(
                self.camera_id,
                self.area_name,
                f"Received empty frame (ID: {frame_id}). Skipping.",
            )
        try:
            frame_copy = frame.copy()
        except Exception as e:
            self.logger.log_warning(
                self.camera_id,
                self.area_name,
                f"Could not copy frame (ID: {frame_id}) using original: {e}",
            )
            frame_copy = frame

        try:
            timestamp = self.timestamp or datetime.now()
            frame_data = FrameData(
                camera_id=self.camera_id,
                frame_id=frame_id,
                image=frame_copy,
                timestamp=timestamp,
            )
        except Exception as e:
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Error initializing FrameData for frame {frame_id}: {e}",
            )
            return None

        try:
            for module in self.modules:
                frame_data = module.process(frame_data)
        except Exception as e:
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Error processing frame {frame_id} with AI modules: {e}",
                exc_info=True,
            )
            return None

        return frame_data

    def _setup_video_writer(self, frame: np.ndarray) -> Optional[cv2.VideoWriter]:
        """Initializes the OpenCV VideoWriter if an output path is specified.

        Uses the frame dimensions, `target_fps`, and `codec` from the
        pipeline's configuration.

        Args:
            frame: A sample frame to get dimensions for the video writer.

        Returns:
            An initialized `cv2.VideoWriter` object, or None if `output_path`
            is not set.
        """
        if not self.output_path:
            return None

        h, w = frame.shape[:2]

        # Make sure we have a valid frame rate for the writer
        fps = self.target_fps
        if fps <= 0:
            # Fallback to default 30fps if we couldn't determine source FPS
            fps = self.frame_rate if self.frame_rate > 0 else 30
            self.logger.log_info(
                self.camera_id, self.area_name, f"Using {fps} fps for video writer"
            )

        # Create the directory if it doesn't exist
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Initialize the writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)  # type: ignore cv2 bug
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

        # Verify writer was created successfully
        if writer is None or not writer.isOpened():
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Failed to create video writer for {self.output_path}. Check codec '{self.codec}' and output path.",
            )
            return None

        return writer

    def _display_output(self, frame_data: FrameData) -> bool:
        """Displays the processed frame if visualization is enabled.

        Renders detections or tracks onto the frame image based on the
        `visualize_type`. If a display environment is available, it shows
        the image in a window. Handles 'q' key press to quit.
        If display fails, it switches to headless mode.

        Args:
            frame_data: The `FrameData` object containing the image and results
                to display.

        Returns:
            True if processing should continue, False if the user quit.
        """
        if not self.visualize or frame_data is None:
            return True

        items, statuses, cls_ids = [], [], []
        if self.visualize_type == "detection":
            items = frame_data.detection
            statuses = [
                f"{getattr(obj, 'label', 'unknown')} {getattr(obj, 'confidence', 0):.2f}"
                for obj in items
            ]
            cls_ids = [getattr(obj, "label", "unknown") for obj in items]

        elif self.visualize_type == "tracking":
            for obj in frame_data.track:
                if getattr(obj, "label", "Unknown").lower() not in ("body", "human"):
                    continue

                track_id = getattr(obj, "track_id", "Unknown")
                status_text = f"ID-{track_id}"
                cls_id = track_id

                items.append(obj)
                statuses.append(status_text)
                cls_ids.append(cls_id)

        bboxes = [d.box.tolist() for d in items]

        frame_data.image = visualize(
            image=frame_data.image,
            bboxes=bboxes,
            class_ids=cls_ids,
            statuses=statuses,
            cmap=self.cmap,
        )

        # Only try to display if a GUI is available
        if self.display:
            try:
                cv2.imshow(
                    "FPT Camera Output",
                    np.ascontiguousarray(frame_data.image, dtype=np.uint8),
                )
                delay = (
                    max(1, int(1000 / self.target_fps)) if self.target_fps > 0 else 1
                )
                key = cv2.waitKey(delay) & 0xFF
                return key != ord("q")
            except Exception as e:
                self.logger.log_warning(
                    self.camera_id,
                    self.area_name,
                    f"Display error: {e}, continuing in headless mode",
                )
                # If display fails, switch to headless mode for future frames
                self.display = False

        # Headless mode - always continue processing
        return True
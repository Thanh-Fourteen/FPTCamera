import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from input.base import InputSource
from input.frame_buffer import (
    LatestFrameStrategy,
    AdaptiveBufferStrategy,
    FixedSizeQueueStrategy,
)
from utils.logger import LoggerManager


IMG_FORMATS = (
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
)  # include image suffixes
VID_FORMATS = (
    "asf",
    "avi",
    "gif",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "ts",
    "wmv",
)  # include video suffixes


CAP_FFMPEG_RTSP_UDP = 0  # OpenCV CAP_PROP_RTSP_TRANSPORT value for UDP
CAP_FFMPEG_RTSP_TCP = 1  # OpenCV CAP_PROP_RTSP_TRANSPORT value for TCP


def check_buffer_strategy(buffer_strategy, buffer_size):
    """
    Select a FrameBuffer strategy based on config.
    """
    if buffer_strategy == "latest":
        return LatestFrameStrategy
    elif buffer_strategy == "adaptive":
        return AdaptiveBufferStrategy(min_size=1, max_size=buffer_size)

    return FixedSizeQueueStrategy(max_size=buffer_size)


class VideoInputSource(InputSource):
    def __init__(
        self,
        rotation_degrees: int,
        camera_id: str,
        area_name: str,
        logger: LoggerManager,
    ):
        super().__init__(rotation_degrees)

        self.logger = logger
        self.camera_id = camera_id
        self.area_name = area_name

        # init
        self._source_path: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened_flag = False
        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0

    def open(self, source_path: str) -> bool:
        """
        Open the given source (file path, RTSP URI, etc.).
        Returns True if opened successfully, False otherwise.
        """
        if source_path.isdigit():
            source_to_open = int(source_path)
        else:
            source_to_open = source_path

        self._cap = cv2.VideoCapture(source_to_open)

        if not self._cap.isOpened():
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Failed to open video source: {source_path}",
            )
            self._is_opened_flag = False

            return False

        self.logger.log_info(
            self.camera_id,
            self.area_name,
            f"Successfully opened video source: {source_path}",
        )
        self._is_opened_flag = True

        # Setup param
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self._fps == 0:
            self._fps = 30  # Set default
            self.logger.log_warning(
                self.camera_id,
                self.area_name,
                f"Could not determine FPS for source {source_path}. Defaulting to {self._fps} FPS.",
            )

        self._frame_width, self._frame_height = self._get_rotated_dimensions(
            self._frame_width, self._frame_height
        )

        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the source.
        Returns a tuple (success_flag, frame_array).
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret:
            return False, None

        rotated_frame = self._rotate_frame(frame)

        return True, rotated_frame

    def is_opened(self) -> bool:
        """
        Check if the source is currently opened.
        """
        return self._is_opened_flag

    def get_fps(self) -> float:
        """
        Get the frames-per-second of the source.
        """
        return self._fps

    def close(self) -> None:
        """
        Close the source and release resources.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._is_opened_flag = False

            self.logger.log_info(
                self.camera_id, " ", f"Closing video source: {self._source_path}"
            )

    def get_frame_count(self) -> int:
        """
        Get the total number of frames in the source (if applicable).
        """
        return self._frame_count

    def get_dimensions(self) -> Tuple[int, int]:  # (width, height)
        """
        Get the frame dimensions (width, height).
        """
        return self._frame_width, self._frame_height


class ImageInputSource(InputSource):
    def __init__(
        self,
        rotation_degrees: int = 0,
        camera_id: str = "None",
        area_name: str = "None",
        logger: LoggerManager = LoggerManager(),
    ):
        super().__init__(rotation_degrees)
        self.camera_id = camera_id
        self.area_name = area_name
        self.logger = logger

        # init
        self._source_path: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened_flag = False
        self._has_been_read = False
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._fps = 1.0

    def open(self, source_path: str) -> bool:
        """
        Open the given source (file path, RTSP URI, etc.).
        Returns True if opened successfully, False otherwise.
        """
        self._source_path = source_path
        image_file_path = Path(source_path)

        if not image_file_path.exists():
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Failed to load image path: {source_path}",
            )
            self._is_opened_flag = False

            return False

        self._frame = cv2.imread(str(image_file_path))
        if self._frame is None:
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Failed to load image path: {source_path}. Check file integrity or format.",
            )
            self._is_opened_flag = False

            return False
        else:
            self.logger.log_info(
                self.camera_id,
                self.area_name,
                f"Successfully loaded image: {source_path}",
            )
            self._is_opened_flag = True
            self._has_been_read = False

            self._frame = self._rotate_frame(self._frame)
            if self._frame is not None:
                self._frame_height, self._frame_width = self._frame.shape[:2]

            return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the source.
        Returns a tuple (success_flag, frame_array).
        """
        if not self._is_opened_flag or self._has_been_read:
            return False, None

        self._has_been_read = True
        return True, self._frame

    def is_opened(self) -> bool:
        """
        Check if the source is currently opened.
        """
        return self._is_opened_flag and self._has_been_read

    def get_fps(self) -> float:
        """
        Get the frames-per-second of the source.
        """
        return self._fps

    def close(self) -> None:
        """
        Close the source and release resources.
        """
        if self._is_opened_flag:
            self.logger.log_info(
                self.camera_id,
                self.area_name,
                f"Image file source {self._source_path} released.",
            )
            self._is_opened_flag = False
            self._frame = None

    def get_frame_count(self) -> int:
        """
        Get the total number of frames in the source (if applicable).
        """
        return self._frame_count

    def get_dimensions(self) -> Tuple[int, int]:  # (width, height)
        """
        Get the frame dimensions (width, height).
        """
        return self._frame_width, self._frame_height


class RTSPInputSource(InputSource):
    def __init__(
        self,
        rotation_degrees: int,
        camera_id: str = "None",
        area_name: str = "None",
        logger: LoggerManager = LoggerManager(),
        transport_protocol: str = "tcp",
    ):
        super().__init__(rotation_degrees)
        self.camera_id = camera_id
        self.area_name = area_name
        self.logger = logger
        self.transport_protocol = transport_protocol.lower()

        # init
        self._url: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened_flag = False
        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0

    def open(self, source_path: str) -> bool:
        """
        Open the RTSP stream.
        """
        transport_map = {
            "tcp": CAP_FFMPEG_RTSP_TCP,
            "udp": CAP_FFMPEG_RTSP_UDP,
        }

        self.logger.log_info(
            self.camera_id,
            self.area_name,
            f"Attempting to open RTSP source: {source_path} with transport: {self.transport_protocol.upper()}",
        )

        rtsp_transport_prop = transport_map.get(self.transport_protocol)
        if rtsp_transport_prop is None:
            self.logger.log_warning(
                self.camera_id,
                self.area_name,
                f"Unsupported transport protocol '{self.transport_protocol}'. Defaulting to TCP.",
            )
            rtsp_transport_prop = CAP_FFMPEG_RTSP_TCP

        # Initialize VideoCapture, explicitly using the FFMPEG backend for RTSP
        self._cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
        try:
            self._cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, rtsp_transport_prop)  # type: ignore
        except Exception:
            pass

        if not self._cap.isOpened():
            self.logger.log_error(
                self.camera_id,
                self.area_name,
                f"Failed to open RTSP source: {source_path} using {self.transport_protocol.upper()}.",
            )
            self._is_opened_flag = False
            return False

        self.logger.log_info(
            self.camera_id,
            self.area_name,
            f"Successfully opened RTSP source: {source_path} using {self.transport_protocol.upper()}.",
        )
        self._is_opened_flag = True
        self._source_path = source_path

        # Setup param
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = -1  # Not applicable for live streams

        if self._fps == 0.0:
            self.logger.log_warning(
                self.camera_id,
                self.area_name,
                f"Could not determine FPS for source {source_path}. Defaulting to {self._fps} FPS.",
            )

        self._frame_width, self._frame_height = self._get_rotated_dimensions(
            self._frame_width, self._frame_height
        )

        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the RTSP source.
        Returns a tuple (success_flag, frame_array).
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if frame is not None and ret:
            frame = self._rotate_frame(frame)

        return ret, frame

    def is_opened(self) -> bool:
        """
        Check if the RTSP source is currently opened.
        """
        return self._is_opened_flag

    def get_fps(self) -> float:
        """
        Get the frames-per-second of the RTSP source.
        """
        return self._fps

    def close(self) -> None:
        """
        Close the RTSP source and release resources.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._is_opened_flag = False

            self.logger.log_info(
                self.camera_id,
                self.area_name,
                f"Closing RTSP source: {self._source_path}",
            )

    def get_frame_count(self) -> int:
        """
        Get the total number of frames in the RTSP source (if applicable).
        """
        return self._frame_count

    def get_dimensions(self) -> Tuple[int, int]:  # (width, height)
        """Get the frame dimensions (width, height)."""
        return self._frame_width, self._frame_height

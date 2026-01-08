"""
Abstract base class for all input sources (file, image, RTSP, etc.).
"""

import abc
import numpy as np
import cv2
from typing import Optional, Tuple


class InputSource(abc.ABC):
    """
    Defines the interface for different input sources.
    """

    def __init__(self, rotation_degrees: int = 0):
        """Initialize input source with rotation configuration.

        Args:
            rotation_degrees: Degrees to rotate frames (0, 90, 180, 270)
        """
        self.rotation_degrees = rotation_degrees

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply rotation to frame if needed.

        Args:
            frame: Input frame to rotate

        Returns:
            Rotated frame
        """
        if self.rotation_degrees == 0 or frame is None:
            return frame

        if self.rotation_degrees == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_degrees == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_degrees == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return frame

    def _get_rotated_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """Get dimensions after rotation.

        Args:
            width: Original width
            height: Original height

        Returns:
            (width, height) after rotation
        """
        if self.rotation_degrees in [90, 270]:
            return height, width  # Swap dimensions
        return width, height

    @abc.abstractmethod
    def open(self, source_path: str) -> bool:
        """
        Open the given source (file path, RTSP URI, etc.).
        Returns True if opened successfully, False otherwise.
        """
        pass

    @abc.abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:  # (success, frame)
        """
        Read the next frame from the source.
        Returns a tuple (success_flag, frame_array).
        """
        pass

    @abc.abstractmethod
    def is_opened(self) -> bool:
        """
        Check if the source is currently opened.
        """
        pass

    @abc.abstractmethod
    def get_fps(self) -> float:
        """
        Get the frames-per-second of the source.
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close the source and release resources.
        """
        pass

    @abc.abstractmethod
    def get_frame_count(self) -> int:
        """
        Get the total number of frames in the source (if applicable).
        """
        pass

    @abc.abstractmethod
    def get_dimensions(self) -> Tuple[int, int]:  # (width, height)
        """
        Get the frame dimensions (width, height).
        """
        pass
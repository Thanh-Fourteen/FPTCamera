import numpy as np
from datetime import datetime
from typing import Union
from pydantic import BaseModel, ConfigDict, Field


# General
class Coordinate(BaseModel):
    """Represents a 2D coordinate point with float precision.

    Attributes:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        confidence (float): The confidence score of the detection (e.g., between 0.0 and 1.0).
    """

    x: float
    y: float
    confidence: float = 1.0


class BoundingBox(BaseModel):
    """Represents a rectangular bounding box with integer coordinates.

    Attributes:
        x1 (int): The x-coordinate of the top-left corner.
        y1 (int): The y-coordinate of the top-left corner.
        x2 (int): The x-coordinate of the bottom-right corner.
        y2 (int): The y-coordinate of the bottom-right corner.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Calculates and returns the width of the bounding box.

        Returns:
            int: The width of the bounding box (x2 - x1).
        """
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Calculates and returns the height of the bounding box.

        Returns:
            int: The height of the bounding box (y2 - y1).

        """
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Calculates and returns the area of the bounding box.

        Returns:
            int: The area of the bounding box (width * height).

        """
        return self.width * self.height

    @property
    def center_x(self) -> float:
        """Calculates and returns the x-coordinate of the center of the bounding box.

        Returns:
            float: The center x-coordinate.
        """
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        """Calculates and returns the y-coordinate of the center of the bounding box.

        Returns:
            float: The center y-coordinate.
        """
        return (self.y1 + self.y2) / 2.0

    def tolist(self) -> list:
        """Converts the bounding box coordinates to a list.

        Returns:
            list: A list of coordinates in the format [x1, y1, x2, y2].
        """
        return [self.x1, self.y1, self.x2, self.y2]

    @staticmethod
    def fromlist(coordinates: Union[list, np.ndarray]) -> "BoundingBox":
        """Create a BoundingBox from a list or numpy array of coordinates.

        Args:
            coordinates (list | np.ndarray): A list or numpy array of four coordinates
                [x1, y1, x2, y2].

        Returns:
            BoundingBox: A new BoundingBox object.

        Raises:
            ValueError: If the coordinates list does not have exactly 4 elements.
        """
        if len(coordinates) != 4:
            raise ValueError("Coordinates must have exactly 4 elements")
        return BoundingBox(
            x1=int(coordinates[0]),
            y1=int(coordinates[1]),
            x2=int(coordinates[2]),
            y2=int(coordinates[3]),
        )


# Detection
class Detection(BaseModel):
    """Data Transfer Object for a single object detection, potentially with sub-detections.

    Attributes:
        box (BoundingBox): The bounding box of the detected object.
        confidence (float): The confidence score of the detection (e.g., between 0.0 and 1.0).
        label (str): The class label of the detected object (default is "None").
        state (str): The current state of the detection (e.g., 'new', 'tracked', 'lost'). Default "unknown"
        keypoints (List[Coordinate]): Optional list of keypoints associated with the detection.
    """

    box: BoundingBox
    confidence: float
    label: str = "None"
    state: str = "unknown"
    keypoints: list[Coordinate] = Field(default_factory=list)


class TrackingObject(Detection):
    """Represents a single tracked object, maintaining its identity across frames.

    Attributes:
        track_id (int): A unique identifier for the tracked object across frames. Defaults to -1
            if unassigned.
        prev_state (TrackState): The current tracking state of the object (e.g., NEW, ACTIVE, LOST, REMOVED).
    """

    track_id: int = Field(default=-1)
    prev_state: str = Field(default="unknown")


class FrameData(BaseModel):
    """A comprehensive Data Transfer Object representing all processed information
    for a single video frame system pipeline.

    Attributes:
        camera_id (str): Identifier for the camera that captured the frame.
            Defaults to "camera_0".
        frame_id (int): Unique sequential identifier of the frame.
        image (np.ndarray): The raw image data (in BGR or RGB format) of the frame.
        timestamp (datetime): The precise date and time when the frame was captured or processed.
        detection (List[Detection]): A list of object detection results associated with this frame.
            Empty if no detections are available.
        track (List[TrackingObject]): A list of tracked objects within the frame.
            Empty if tracking has not been performed or no tracks are available.
    """

    camera_id: str = Field(default="camera_0")
    frame_id: int
    image: np.ndarray
    timestamp: datetime
    detection: list[Detection] = Field(default_factory=list)
    track: list[TrackingObject] = Field(default_factory=list)

    # Even layer
    crowd_count: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

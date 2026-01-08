import importlib.metadata as metadata

from components.detection.base import BaseDetection
from src.components.detection.yolo import HumanDetection

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = ["__version__", "BaseDetection", "HumanDetection"]

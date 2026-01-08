import importlib.metadata as metadata

from components.tracking.base import BaseTracking
from components.tracking.ByteTrack.track import ByteTrack

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "__version__",
    "BaseTracking",
    "ByteTrack"
]
from abc import ABC, abstractmethod
from typing import Any, Dict

from utils.dtos import FrameData
from utils.logger import LoggerManager


class IAIModule(ABC):
    """Abstract interface for AI modules.

    All AI modules must implement these methods to integrate with the pipeline.
    """

    @abstractmethod
    def setup(self, config: Dict[str, Any], logger: LoggerManager) -> None:
        """Initialize module with configuration and logger.

        Args:
            config (Dict[str, Any]): Module-specific parameters.
            logger (Logger): Logger instance for event recording.
        """
        pass

    @abstractmethod
    def process(self, frame_data: FrameData) -> FrameData:
        """Process a frame and return the modified data.

        Args:
            frame_data (FrameData): Input frame data.

        Returns:
            FrameData: Processed frame data.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this AI module."""
        pass


class BaseAIModule(IAIModule):
    """Base class providing common functionality to AI modules.

    Attributes:
        camera_id (str): Camera source identifier.
        metadata (Dict[str, Any]): Module metadata for logs.
    """

    def __init__(self) -> None:
        """Initialize the BaseAIModule."""
        self._logger: LoggerManager
        self._name: str = self.__class__.__name__
        self._config: Dict[str, Any] = {}
        self.camera_id: str = "unknown"
        self.area_name: str = "unknown_area"
        self.metadata: Dict[str, Any] = {}

    def setup(self, config: Dict[str, Any], logger: LoggerManager) -> None:
        """Store config, extract metadata, and log setup.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            logger (Logger): Logger for recording setup events.
        """
        # store config and logger
        self._config = config
        self._logger = logger

        # centralize common metadata
        self.camera_id = config.get("camera_id", "unknown")
        self.area_name = config.get("area_name", "unknown_area")
        self.metadata = {
            "module_name": self._name,
        }

        # log setup with consistent metadata
        self._logger.log_info(
            self.camera_id,
            self.area_name,
            f"Module {self._name} set up with config: {self._config}",
        )

    @property
    def config(self) -> Dict[str, Any]:
        """Current module configuration."""
        return self._config

    @property
    def name(self) -> str:
        """Module name."""
        return self._name

    @property
    def logger(self) -> LoggerManager:
        """Logger instance for the module."""
        return self._logger

    def log_module_info(self, message: str) -> None:
        """Log an informational message with module metadata.

        Args:
            message (str): Information to log.
        """
        self._logger.log_info(
            self.camera_id,
            self.area_name,
            f"[{self._name}] {message}",
        )

    def log_module_error(self, message: str, exc_info: bool = False) -> None:
        """Log an error message with module metadata.

        Args:
            message (str): Error description.
            exc_info (bool): Include exception info if True.
        """
        self._logger.log_error(
            self.camera_id,
            self.area_name,
            f"[{self._name}] {message}",
            exc_info=exc_info,
        )

    @abstractmethod
    def process(self, frame_data: FrameData) -> FrameData:
        """Processes the given frame data. This method must be implemented by subclasses.

        Args:
            frame_data (FrameData): The input frame data to be processed.

        Returns:
            FrameData: The processed frame data.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

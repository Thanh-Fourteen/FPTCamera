from abc import ABC, abstractmethod
from typing import Any, Dict

from utils.logger import LoggerManager


class IStrategy(ABC):
    """
    Interface for strategy classes.
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any], logger: LoggerManager) -> None:
        """
        Set up the strategy with the given configuration and logger.

        Args:
            config (Dict[str, Any]): Configuration for the strategy.
            logger (LoggerManager): Logger instance for logging.
        """
        pass

    @abstractmethod
    def execute(self, data: Any) -> Any:
        """
        Execute the strategy on the given data.

        Args:
            data (Any): Data to be processed by the strategy.

        Returns:
            Any: Result of the strategy execution.
        """
        pass
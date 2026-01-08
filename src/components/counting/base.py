"""Counting strategy interface for SmartCity pipeline.

Defines ICountingStrategy requiring `execute(data) -> int`.
"""

from abc import abstractmethod
from typing import Dict

from utils.dtos import FrameData
from components.base_strategy import IStrategy


class BaseCounting(IStrategy):
    """Abstract interface for counting strategies.

    Methods:
        execute(FrameData) -> int: Perform object counting on frame data.
    """

    @abstractmethod
    def execute(self, data: FrameData) -> Dict[str, int]:
        """
        Perform counting on the given frame data.

        Args:
            data (FrameData): The frame data to be processed.

        Returns:
            Dict[str, int]: A dictionary mapping class names to their counts.
        """
        pass

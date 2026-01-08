from typing import Dict, Any, Optional, Type, TypedDict

from components.base_strategy import IStrategy
from components.detection import BaseDetection, HumanDetection
from components.tracking import BaseTracking, ByteTrack
from components.counting import BaseCounting, PolygonDirectionalCounter

from modules.detector import Detector
from modules.tracker import Tracker
from modules.crowd_counter import CrowdCounter
from core.base_module import BaseAIModule
from utils.logger import LoggerManager


class ModuleMetadata(TypedDict):
    """Metadata for each module, including its class, strategy interface, and default strategy."""

    class_: Type[BaseAIModule]
    strategy_interface: Optional[Type[IStrategy]]
    default_strategy: Optional[str]


class FactoryModule:
    """
    A factory class responsible for creating and managing AI modules and their associated strategies.
    """

    def __init__(self) -> None:
        """Initializes the FactoryModule with predefined strategy and module metadata registries.

        Attributes:
            _strategy_registry (Dict[str, Type[IStrategy]]): A dictionary mapping lowercase
                strategy names to their respective class types.
            _module_metadata (Dict[str, ModuleMetadata]): A dictionary mapping lowercase
                module names to their metadata, including the module class, its required
                strategy interface, and a default strategy name.
        """
        self._strategy_registry = {
            "deyo_hfb": HumanDetection,
            "bytetrack": ByteTrack,
            "humancount": PolygonDirectionalCounter,
        }

        self._module_metadata: Dict[str, ModuleMetadata] = {
            "human_detector": {
                "class_": Detector,
                "strategy_interface": BaseDetection,
                "default_strategy": "deyo_hfb",
            },
            "tracker": {
                "class_": Tracker,
                "strategy_interface": BaseTracking,
                "default_strategy": "bytetrack",
            },
            "human_counter": {
                "class_": CrowdCounter,
                "strategy_interface": BaseCounting,
                "default_strategy": "humancount",
            },
        }

    def create_strategy(
        self, strategy_name: str, config: Dict[str, Any], logger: LoggerManager
    ):
        """Creates an instance of a strategy based on its name and configuration.

        Args:
            strategy_name (str): Case-insensitive name of the strategy.
            config (Dict[str, Any]): A dictionary containing configuration parameters for the strategy.
            logger (LoggerManager): An instance of LoggerManager for logging messages.

        Returns:
            An instance of the strategy if successful, otherwise None.
        """
        self.camera_id = config.get("camera_id", "unknown")
        self.area_name = config.get("area_name", "unknown_area")
        strategy_class = self._strategy_registry.get(strategy_name.lower())
        if strategy_class:
            try:
                instance = strategy_class()
                instance.initialize(config, logger)
                logger.log_debug(
                    self.camera_id, self.area_name, f"Created strategy: {strategy_name}"
                )

                return instance

            except Exception as e:
                logger.log_error(
                    self.camera_id,
                    self.area_name,
                    f"Failed to create strategy {strategy_name}: {e}",
                    exc_info=True,
                )

                return None

    def create_module(
        self, module_name: str, config: Dict[str, Any], logger: LoggerManager
    ):
        """Creates an instance of an AI module based on its name and configuration.

        Args:
            module_name (str): The name of the module to create (case-insensitive).
            config (Dict[str, Any]): Configuration dictionary for the module and strategy.
            logger (LoggerManager): An instance of LoggerManager for logging messages.

        Returns:
            The initialized module instance, or None on failure.
        """
        metadata = self._module_metadata.get(module_name.lower())
        self.camera_id = config.get("camera_id", "unknown")
        self.area_name = config.get("area_name", "unknown_area")

        if not metadata:
            logger.log_error(
                self.camera_id,
                self.area_name,
                f"Module name {module_name} not found",
                exc_info=True,
            )
            return None

        try:
            module_class = metadata["class_"]
            strategy_interface = metadata.get("strategy_interface")
            strategy_name = metadata.get("default_strategy")

            if strategy_interface and strategy_name:
                strategy = self.create_strategy(strategy_name, config, logger)

                if not strategy or not isinstance(strategy, strategy_interface):
                    logger.log_error(
                        "Factory",
                        module_name,
                        f"Invalid strategy '{strategy_name}' for module '{module_name}'.",
                    )
                    return None
                module_instance = module_class(strategy)
            else:
                module_instance = module_class()

            module_instance.setup(config=config, logger=logger)
            logger.log_debug("System", " ", f"Created module: {module_name}")

            return module_instance

        except Exception as e:
            logger.log_error(
                self.camera_id,
                self.area_name,
                f"Failed to create module '{module_name}': {e}",
                exc_info=True,
            )

            return None

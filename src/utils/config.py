"""Manages loading and accessing configuration settings from YAML files.

This module provides a singleton class `ConfigManager` that handles loading
configuration from a primary YAML file and any other files it includes.
It supports hierarchical configurations and deep merging of settings.
"""

import os
import threading
from typing import Any, Dict, Optional

import yaml

from utils.logger import LoggerManager


class ConfigManager:
    """
    Singleton class to manage configuration settings from a YAML file.

    This class provides a centralized way to load and access application
    configuration. It supports including other YAML files within a main
    configuration file and deep-merges them, with later definitions
    overriding earlier ones.

    Attributes:
        _instance (Optional[ConfigManager]): The singleton instance of the class.
        _lock (threading.Lock): A lock to ensure thread-safe instantiation.
        _config_data (Optional[Dict[str, Any]]): Stores the loaded configuration.
        _config_path (Optional[str]): Path to the main configuration file.
        _loaded_files (Set[str]): A set to keep track of loaded files to prevent
                                 circular includes.
        logger (LoggerManager): An instance of the Logger for logging messages.
    """

    _instance = None
    _lock = threading.Lock()
    _config_data: Optional[Dict[str, Any]] = None
    _config_path: Optional[str] = None
    _loaded_files = set()

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        """
        Gets the singleton instance of ConfigManager.

        Ensures that only one instance of ConfigManager is created and used
        throughout the application.

        Returns:
            ConfigManager: The singleton instance of ConfigManager.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    cls._instance.__init__()
        return cls._instance

    def __init__(self):
        """Initializes the ConfigManager.

        Sets up the logger and initializes the set of loaded files.
        Note: This is called when the first instance is created by `get_instance`.
        """
        self.logger = LoggerManager.get_instance()
        self._loaded_files = set()

    def load_config(self, config_path: str) -> None:
        """
        Loads configuration from a YAML file and its includes.

        The main entry point for loading configuration. It resets any previously
        loaded configuration and starts the loading process from the specified
        `config_path`.

        Args:
            config_path (str): Path to the primary YAML configuration file.

        Raises:
            FileNotFoundError: If the `config_path` or any included file does not exist.
            yaml.YAMLError: If there is an error parsing any YAML file.
            Exception: For other unexpected errors during loading.
        """
        with self._lock:
            self._loaded_files = set()  # Reset loaded files for a new load
            self._config_path = config_path
            self._config_data = self._load_inherited(config_path)
            print(f"Configuration loaded successfully from {config_path}")

    def _load_inherited(self, file_path: str) -> Dict[str, Any]:
        """
        Loads a YAML file and recursively resolves its 'includes'.

        This method handles the actual file loading and parsing. If an 'includes'
        key is found, it recursively calls itself to load and merge those files first.
        It also detects and prevents circular includes.

        Args:
            file_path (str): Path to the YAML file to load.

        Returns:
            Dict[str, Any]: The loaded and merged configuration from this file
                            and its includes.

        Raises:
            FileNotFoundError: If `file_path` does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        # Convert to absolute path for tracking loaded files
        abs_path = os.path.abspath(file_path)

        # Avoid circular includes
        if abs_path in self._loaded_files:
            print(f"Circular include detected for {file_path}, skipping")
            return {}

        self._loaded_files.add(abs_path)

        try:
            with open(file_path, "r") as file:
                config = yaml.safe_load(file) or {}

            # Handle includes if present
            if "includes" in config:
                includes = config.pop("includes")
                if includes:
                    # Get the directory of the current file to resolve relative paths
                    base_dir = os.path.dirname(file_path)

                    # Process each include
                    for include in includes:
                        # Resolve path relative to the current file
                        include_path = os.path.join(base_dir, include)

                        # Load and merge the included config
                        include_config = self._load_inherited(include_path)

                        # Deep merge the configs
                        config = self._deep_merge(include_config, config)

            return config

        except FileNotFoundError:
            print(f"Configuration file not found: {file_path}")
            raise
        except yaml.YAMLError as e:
            print(f"Error parsing {file_path}: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error loading config {file_path}: {e}")
            raise

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merges two dictionaries, with 'override' taking precedence.

        If a key exists in both dictionaries and both values are dictionaries,
        it recursively merges them. Otherwise, the value from `override` is used.

        Args:
            base (Dict[str, Any]): The base dictionary.
            override (Dict[str, Any]): The dictionary whose values will override
                                     those in `base`.

        Returns:
            Dict[str, Any]: The merged dictionary.
        """
        result = base.copy()

        for key, value in override.items():
            # If both values are dictionaries, recursively merge them
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                # Otherwise, override or add the value
                result[key] = value

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using dot notation for nested keys.

        Example:
            `config_manager.get("database.host", "localhost")`

        Args:
            key (str): The dot-separated key (e.g., "application.name").
            default (Any, optional): The default value to return if the key is
                                     not found. Defaults to None.

        Returns:
            Any: The configuration value if found, otherwise the `default` value.

        Raises:
            ValueError: If `load_config` has not been called successfully before
                        calling `get`.
        """
        if self._config_data is None:
            raise ValueError("Configuration data is not loaded.")

        keys = key.split(".")
        value = self._config_data

        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    print(
                        f"Config key path '{key}' leads to a non-dict value at '{k}'."
                    )
                    return default
            return value
        except KeyError:
            return default
        except Exception as e:
            print(f"Error accessing config key '{key}': {e}")
            return default
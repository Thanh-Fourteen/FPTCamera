import csv
import sys
import logging
import logging.handlers
import threading

from typing import Optional, Dict, Any
from pathlib import Path


DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FMT = "%Y-%m-%d %H:%M:%S"
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def resolve_log_level(level: Optional[str]) -> int:
    """Convert string level to logging level constant"""
    if not level:
        return logging.INFO
    return LOG_LEVELS.get(level.lower(), logging.INFO)


def get_console_handler(
    level: str, format: Optional[str], date_format: Optional[str]
) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(resolve_log_level(level))
    handler.setFormatter(
        logging.Formatter(
            fmt=format or DEFAULT_FORMAT, datefmt=date_format or DEFAULT_DATE_FMT
        )
    )

    return handler


def get_rotating_file_handler(
    filepath: str,
    level: str,
    format: Optional[str],
    date_format: Optional[str],
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Handler:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        filepath, maxBytes=max_bytes, backupCount=backup_count
    )
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler.setFormatter(
        logging.Formatter(
            fmt=format or DEFAULT_FORMAT, datefmt=date_format or DEFAULT_DATE_FMT
        )
    )

    return handler


class CSVLogHandler(logging.Handler):
    def __init__(self, filename: str, fields: Optional[list] = None):
        super().__init__()
        self.filename = filename
        self.fields = fields
        self._lock = threading.Lock()

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        if self.fields and not Path(filename).exists():
            with open(filename, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with (
                self._lock,
                open(self.filename, mode="a", newline="", encoding="utf-8") as f,
            ):
                if self.fields and isinstance(record.msg, dict):
                    writer = csv.DictWriter(f, fieldnames=self.fields)
                    writer.writerow(record.msg)
                else:
                    writer = csv.writer(f)
                    writer.writerow([self.format(record)])
        except Exception:
            self.handleError(record)


class LoggerManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._logger: Optional[logging.Logger] = None
            self._specialized_loggers: Dict[str, logging.Logger] = {}
            self.config_manager: Optional[Any] = None  # Khởi tạo là None
            self._initialized = True

    @classmethod
    def get_instance(cls) -> "Logger":
        """
        Gets the singleton instance of the Logger.

        This is the public method to access the Logger instance.

        Returns:
            Logger: The singleton Logger instance.
        """
        return cls()

    def setup_logging(
        self,
        config_manager,
        project_name: str,
        log_level: str = "info",
        log_file: Optional[str] = None,
        format: Optional[str] = None,
        date_format: Optional[str] = None,
    ):
        if self._logger is not None:
            print("LoggerManager initialize.", file=sys.stderr)
            return

        self.config_manager = config_manager
        self._logger = self._initialize_main_logger(
            project_name,
            log_level,
            log_file,
            format,
            date_format,
        )
        self._initialize_specialized_loggers(project_name, format, date_format)

    def _initialize_main_logger(
        self,
        project_name: str,
        log_level: str = "info",
        log_file: Optional[str] = None,
        format: Optional[str] = None,
        date_format: Optional[str] = None,
    ):
        logger = logging.getLogger(project_name)
        logger.setLevel(resolve_log_level(log_level))
        logger.handlers.clear()

        # handler
        logger.addHandler(get_console_handler(log_level, format, date_format))
        if log_file:
            logger.addHandler(
                get_rotating_file_handler(log_file, log_level, format, date_format)
            )

        return logger

    def _initialize_specialized_loggers(self, project_name, format, date_format):
        if not self.config_manager:
            sys.stderr.write(
                "ConfigManager not initialize. Can not create specialized logger.\n"
            )
            return

        log_config = self.config_manager.get("logging", {})
        debug_mode = log_config.get("debug_mode", False)
        specialized_config = log_config.get("specialized_loggers", {})

        if not debug_mode:
            return

        for logger_name, logger_config in specialized_config.items():
            normalized_logger_name = logger_name.lower()

            try:
                logger = logging.getLogger(f"{project_name}.{logger_name}")
                level = logger_config.get("level", "info")
                logger.setLevel(resolve_log_level(level))
                logger.handlers.clear()

                log_file = logger_config.get("file")
                if log_file:
                    if log_file.endswith(".csv"):
                        fields = logger_config.get("fields")
                        logger.addHandler(CSVLogHandler(log_file, fields=fields))
                    else:
                        logger.addHandler(
                            get_rotating_file_handler(
                                log_file, level, format, date_format
                            )
                        )

                self._specialized_loggers[normalized_logger_name] = logger

            except Exception as e:
                sys.stderr.write(
                    f"ERROR: Failed to initialize specialized logger '{logger_name}': {e}\n"
                )

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        if self._logger is None:
            raise RuntimeError("Logger is not initialized. Call setup_logging first.")
        if name:
            normalized_name = name.lower()
            if normalized_name in self._specialized_loggers:
                return self._specialized_loggers[normalized_name]
            return self._logger.getChild(name)

        return self._logger

    def log_info(
        self, camera_id: Optional[str], area_name: Optional[str], message: str
    ) -> None:
        if self._logger:
            formatted_message = f"[{camera_id}|{area_name}] {message}"
            self._logger.info(formatted_message)

    def log_warning(
        self, camera_id: str, area_name: Optional[str], message: str
    ) -> None:
        """Log a warning message."""
        if self._logger:
            formatted_message = f"[{camera_id}|{area_name}] {message}"
            self._logger.warning(formatted_message)

    def log_debug(self, camera_id: str, area_name: Optional[str], message: str) -> None:
        """Log a debug message."""
        debug_logger = self._specialized_loggers.get("debug")
        if debug_logger:
            formatted_message = f"[{camera_id}|{area_name}] {message}"
            debug_logger.debug(formatted_message)
        elif self._logger:
            formatted_message = f"[{camera_id}] DEBUG: {message}"
            self._logger.debug(formatted_message)

    def log_error(
        self,
        camera_id: Optional[str],
        area_name: Optional[str],
        message: str,
        exc_info: bool = False,
    ) -> None:
        error_logger = self._specialized_loggers.get("error")
        if error_logger:
            formatted_message = (
                f"[{camera_id}|{area_name}] {message}" if camera_id else message
            )
            error_logger.error(formatted_message, exc_info=exc_info)
        elif self._logger:
            formatted_message = (
                f"[{camera_id}] ERROR: {message}" if camera_id else f"ERROR: {message}"
            )
            self._logger.error(formatted_message, exc_info=exc_info)

    def log_critical(
        self,
        camera_id: Optional[str],
        area_name: Optional[str],
        message: str,
        exc_info: bool = False,
    ) -> None:
        if self._logger:
            formatted_message = (
                f"[{camera_id}|{area_name}] {message}" if camera_id else message
            )
            self._logger.critical(formatted_message, exc_info=exc_info)

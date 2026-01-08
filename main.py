import sys
from pathlib import Path
from typing import List

# Add src to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

import argparse
from typing import Tuple, Dict, Any
from src.utils.logger import LoggerManager
from src.utils.config import ConfigManager
from src.core.factory_module import FactoryModule
from src.core.base_module import IAIModule
from src.pipeline.pipeline import Pipeline


def setup_environment(
    config_path: str,
) -> Tuple[ConfigManager, LoggerManager, Dict[str, Any]]:
    config_manager = ConfigManager.get_instance()
    config_manager.load_config(config_path)

    # Get application defaults
    app_config = config_manager.get("application", {})
    default_camera_id = app_config.get("camera_id", "unknown_camera")
    default_area_name = app_config.get("area_name", "unknown_area")

    media_config = app_config.get("input", {})
    media_source = media_config.get("source", None)

    # Logger
    log_config = config_manager.get("logging", {})
    logger = LoggerManager()
    logger.setup_logging(
        config_manager=config_manager,
        project_name=app_config.get("app_name"),
        log_level=log_config.get("default_level", "INFO"),
        log_file=log_config.get("main_log_file", None),
        format=log_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        date_format=log_config.get("date_format", "%Y-%m-%d %H:%M:%S"),
    )
    logger.log_info(default_camera_id, default_area_name, "Logger initialized.")
    logger.log_info(
        default_camera_id,
        default_area_name,
        f"Configuration loaded successfully from: {config_path}",
    )

    return config_manager, logger, media_source


def initialize_modules(config_manager: ConfigManager, logger: LoggerManager):
    module_factory = FactoryModule()

    # Create Modules
    modules_config: List[str] = config_manager.get("modules")
    active_modules: List[IAIModule] = []

    camera_id = config_manager.get("application.camera_id", "unknown")
    area_name = config_manager.get("application.area_name", "unknown")
    logger.log_info(
        camera_id, area_name, f"Initializing {len(modules_config)} modules..."
    )

    for module_name in modules_config:
        if config_manager.get(f"pipeline.{module_name}", False) is not True:
            logger.log_info(
                camera_id,
                area_name,
                f" -> Module '{module_name}' is disabled in pipeline config. Skipping.",
            )
            continue

        module_config = config_manager.get(f"modules.{module_name}", {})
        module_config["camera_id"] = camera_id
        module_config["area_name"] = area_name
        module_instance = module_factory.create_module(
            module_name=module_name,
            config=module_config,
            logger=logger,
        )

        if module_instance:
            active_modules.append(module_instance)
            logger.log_info(
                camera_id,
                area_name,
                f"Module '{module_name}' initialized successfully.",
            )
        else:
            logger.log_warning(
                camera_id,
                area_name,
                f"Module '{module_name}' could not be created. Skipping.",
            )

    logger.log_info(
        camera_id,
        area_name,
        f"Successfully initialized {len(active_modules)}/{len(modules_config)} modules.",
    )
    return active_modules


def main(config_path: str):
    try:
        config_manager, logger, media_source = setup_environment(config_path)
    except Exception as e:
        print(
            f"CRITICAL ERROR: Failed to initialize config or logger: {e}",
            file=sys.stderr,
        )
        # Use default exit code if config manager is not available
        sys.exit(1)

    camera_id = config_manager.get(
        "application.camera_id",
        "unknown_camera",  # set default
    )
    area_name = config_manager.get(
        "application.area_name",
        "unknown_area",  # set default
    )

    try:
        active_modules = initialize_modules(config_manager, logger)
        if not active_modules:
            logger.log_warning(
                camera_id,
                area_name,
                "No active modules found. Pipeline will not perform any processing.",
            )

        pipeline = Pipeline(
            config=config_manager, logger=logger, modules=active_modules
        )
        logger.log_info(
            camera_id,
            area_name,
            f"Pipeline initialized with media source: {media_source}",
        )

        pipeline.run(media_source)
        logger.log_info(
            camera_id, area_name, "Pipeline execution completed successfully."
        )

    except Exception as e:
        logger.log_error(
            camera_id,
            area_name,
            f"CRITICAL ERROR: Failed to run pipeline: {e}",
            exc_info=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", default="./configs/base.yaml", help="Setup config running path"
    )

    opt = parser.parse_args()

    main(**vars(opt))

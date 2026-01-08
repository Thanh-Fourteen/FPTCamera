import os
import asyncio
import logging
import cv2
from typing import List
import sys
from pathlib import Path

from av import VideoFrame
from aiortc import VideoStreamTrack

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from src.pipeline.pipeline import Pipeline
from src.core.base_module import IAIModule
from src.utils.logger import LoggerManager
from src.utils.config import ConfigManager
from src.core.factory_module import FactoryModule

# Configure logging
logger = logging.getLogger(__name__)

# Set OpenCV environment variables for better RTSP handling
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class RTSPVideoStreamTrack(VideoStreamTrack):
    """
    Reads video frames from an RTSP stream using OpenCV and provides them as
    VideoStreamTrack for WebRTC.
    """
    def __init__(self, rtsp_url: str, config_path: str = "./configs/base.yaml"):
        super().__init__()
        self.rtsp_url = rtsp_url
        self._frame_id = 0
        self.cap = None
        self._is_running = True
        self.config = ConfigManager.get_instance()
        self.config.load_config(config_path)    

        self.logger = LoggerManager()
        self.logger.setup_logging(
            config_manager=self.config,
            project_name=self.config.get("application.app_name", "SmartCity"),
            log_level=self.config.get("logging.default_level", "WARNING"),
            log_file=self.config.get("logging.main_log_file", None),
            format=self.config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            date_format=self.config.get("logging.date_format", "%Y-%m-%d %H:%M:%S"),
        )
        self.modules = self._initialize_modules()

        self.pipeline = Pipeline(
            config=self.config,
            logger=self.logger,
            modules=self.modules
        )
        self._open_capture()


    def _initialize_modules(self) -> List[IAIModule]:
        """Initialize modules based on pipeline flags and modules mapping."""
        module_factory = FactoryModule()
        modules_config_map = self.config.get("modules", {}) or {}
        pipeline_flags = self.config.get("pipeline", {}) or {}
        active_module_names: List[str] = [
            name for name, enabled in pipeline_flags.items()
            if bool(enabled) and name in modules_config_map
        ]
        active_modules: List[IAIModule] = []

        camera_id = self.config.get("application.camera_id", "unknown_camera")
        area_name = self.config.get("application.area_name", "unknown_area")
        self.logger.log_info(
            camera_id, area_name, f"Initializing {len(active_module_names)} modules..."
        )

        for module_name in active_module_names:
            module_config = modules_config_map.get(module_name, {})
            module_config["camera_id"] = camera_id
            module_config["area_name"] = area_name
            module_instance = module_factory.create_module(
                module_name=module_name,
                config=module_config,
                logger=self.logger,
            )

            if module_instance:
                active_modules.append(module_instance)
                self.logger.log_info(
                    camera_id,
                    area_name,
                    f"Module '{module_name}' initialized successfully.",
                )
            else:
                self.logger.log_warning(
                    camera_id,
                    area_name,
                    f"Module '{module_name}' could not be created. Skipping.",
                )

        self.logger.log_info(
            camera_id,
            area_name,
            f"Successfully initialized {len(active_modules)}/{len(active_module_names)} modules.",
        )
        if not active_modules:
            self.logger.log_error(
                camera_id,
                area_name,
                "No active modules found. Pipeline may fail.",
            )
            raise ValueError("No valid modules specified in config.")

        return active_modules
    
    def _open_capture(self):
        """Initializes or re-initializes the OpenCV VideoCapture."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        logger.info(f"Attempting to open RTSP stream: {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        backend = self.cap.getBackendName() if hasattr(self.cap, "getBackendName") else "unknown"
        logger.info(f"OpenCV backend used for {self.rtsp_url}: {backend}")

        if not self.cap.isOpened():
            logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
            raise ConnectionError(f"Cannot open RTSP stream: {self.rtsp_url}")
        logger.info(f"Successfully opened RTSP stream: {self.rtsp_url}")

    async def recv(self) -> VideoFrame:
        """
        Reads a single video frame from the RTSP stream and returns it as an av.VideoFrame.
        Handles re-opening the capture if it fails.
        """
        while self._is_running:
            try:
                pts, time_base = await self.next_timestamp()
                success, frame = self.cap.read()

                if not success:
                    logger.warning(f"Failed to read frame from {self.rtsp_url}. Attempting to re-open stream.")
                    self._open_capture() # Try to re-open the capture
                    await asyncio.sleep(0.5) # Give some time before next read attempt
                    continue # Try reading again
                
                processed_frame = await self.pipeline.process_frame(
                    frame, self._frame_id
                )
                self._frame_id += 1
                
                # Convert BGR to RGB for av.VideoFrame
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                video_frame.pts = pts
                video_frame.time_base = time_base
                return video_frame
            
            except ConnectionError as e:
                logger.error(f"RTSP stream error for {self.rtsp_url}: {e}. Stopping track.")
                self._is_running = False
                raise # Re-raise to propagate the error up
            except Exception as e:
                logger.error(f"Unexpected error in RTSPVideoStreamTrack.recv for {self.rtsp_url}: {e}")
                await asyncio.sleep(0.1) # Prevent busy-loop on persistent errors
                # Optionally, attempt re-opening here too if errors are frequent

    async def close(self):
        """Releases the OpenCV VideoCapture resource."""
        logger.info(f"Closing RTSPVideoStreamTrack for {self.rtsp_url}")
        self._is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        await super().close()
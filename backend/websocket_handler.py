import cv2
import base64
import json
import asyncio
import logging
import os
import sys
from typing import List

from aiohttp import web, WSMsgType

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from src.pipeline.pipeline import Pipeline
from src.core.base_module import IAIModule
from src.utils.logger import LoggerManager
from src.utils.config import ConfigManager
from src.core.factory_module import FactoryModule
from src.utils.logger import LoggerManager
from src.utils.config import ConfigManager

logger = logging.getLogger(__name__)

clients = {} # To keep track of active WebSocket connections

class RTSPWebSocketFrameReader:
    """
    Reads video frames from an RTSP stream using OpenCV for WebSocket streaming.
    Includes robust error handling and re-opening logic.
    """
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None
        self._is_running = True
        self._open_capture()

    def _open_capture(self):
        """Initializes or re-initializes the OpenCV VideoCapture."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        logger.info(f"Attempting to open RTSP stream for WebSocket: {self.rtsp_url}")
        # Using CAP_FFMPEG explicitly for RTSP
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        backend = self.cap.getBackendName() if hasattr(self.cap, "getBackendName") else "unknown"
        logger.info(f"OpenCV backend used for {self.rtsp_url} (WebSocket): {backend}")

        if not self.cap.isOpened():
            logger.error(f"Failed to open RTSP stream for WebSocket: {self.rtsp_url}")
            raise ConnectionError(f"Cannot open RTSP stream for WebSocket: {self.rtsp_url}")
        logger.info(f"Successfully opened RTSP stream for WebSocket: {self.rtsp_url}")

    def read_frame(self):
        """Reads a single frame, handling re-opening on failure."""
        success, frame = self.cap.read()
        if not success:
            logger.warning(f"Failed to read frame from {self.rtsp_url} (WebSocket). Attempting to re-open stream.")
            try:
                self._open_capture()
                success, frame = self.cap.read() # Try reading again immediately after re-opening
            except ConnectionError as e:
                logger.error(f"Failed to re-open RTSP stream for WebSocket: {e}")
                self._is_running = False
                raise
        return success, frame

    def stop(self):
        """Stops the frame reader and releases resources."""
        logger.info(f"Stopping RTSPWebSocketFrameReader for {self.rtsp_url}")
        self._is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

async def websocket_handler(request: web.Request, config_path: str = "./configs/base.yaml"):

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    session_id = id(ws)
    clients[session_id] = ws
    logger.info(f"[WS] Client connected: {session_id}")

    # Initialize ConfigManager and LoggerManager
    config_manager = ConfigManager.get_instance()
    config_manager.load_config(config_path)
    logger_manager = LoggerManager()
    logger_manager.setup_logging(
        config_manager=config_manager,
        project_name=config_manager.get("application.app_name", "SmartCity"),
        log_level=config_manager.get("logging.default_level", "INFO"),
        log_file=config_manager.get("logging.main_log_file", None),
        format=config_manager.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        date_format=config_manager.get("logging.date_format", "%Y-%m-%d %H:%M:%S"),
    )

    # Initialize modules using FactoryModule
    module_factory = FactoryModule()
    modules_map = config_manager.get("modules", {}) or {}
    pipeline_flags = config_manager.get("pipeline", {}) or {}
    active_module_names: List[str] = [
        name for name, enabled in pipeline_flags.items()
        if bool(enabled) and name in modules_map
    ]
    active_modules: List[IAIModule] = []
    camera_id = config_manager.get("application.camera_id", "unknown_camera")
    area_name = config_manager.get("application.area_name", "unknown_area")

    for module_name in active_module_names:
        module_config = modules_map.get(module_name, {})
        module_config["camera_id"] = camera_id
        module_config["area_name"] = area_name
        module_instance = module_factory.create_module(
            module_name=module_name,
            config=module_config,
            logger=logger_manager,
        )
        if module_instance:
            active_modules.append(module_instance)
            logger_manager.log_info(
                camera_id,
                area_name,
                f"Module '{module_name}' initialized successfully.",
            )
        else:
            logger_manager.log_warning(
                camera_id,
                area_name,
                f"Module '{module_name}' could not be created. Skipping.",
            )

    if not active_modules:
        logger_manager.log_error(
            camera_id,
            area_name,
            "No active modules found. Pipeline may fail.",
        )
        await ws.send_str(json.dumps({"error": "No active modules found in configuration."}))
        return ws

    pipeline = Pipeline(
        config=config_manager,
        logger=logger_manager,
        modules=active_modules
    )
    frame_reader = None
    stream_task = None
    frame_id = 0

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                rtsp_url = data.get("source")
                logger.info(f"[WS] Client {session_id} requested stream: {rtsp_url}")

                if not rtsp_url:
                    await ws.send_str(json.dumps({"error": "Missing 'source' parameter."}))
                    continue

                # Stop any existing stream for this client
                if stream_task and not stream_task.done():
                    stream_task.cancel()
                    await asyncio.gather(stream_task, return_exceptions=True) # Wait for cancellation
                if frame_reader:
                    frame_reader.stop()

                try:
                    frame_reader = RTSPWebSocketFrameReader(rtsp_url)
                    
                    async def stream_video():
                        nonlocal frame_id
                        while not ws.closed and frame_reader._is_running:
                            try:
                                success, frame = await asyncio.to_thread(frame_reader.read_frame)
                                if not success:
                                    # Frame read failed, _open_capture would have been attempted
                                    # If it's still not running, break loop
                                    if not frame_reader._is_running:
                                        logger.warning(f"Frame reader for {rtsp_url} stopped, ending stream.")
                                        break
                                    await asyncio.sleep(0.1) # Wait a bit before trying again
                                    continue
                                
                                # Process frame using Pipeline
                                processed_frame = await pipeline.process_frame(frame, frame_id)
                                frame_id += 1
                                
                                # Encode to JPEG
                                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]) # Adjust quality as needed
                                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                                await ws.send_str(jpg_as_text)
                                await asyncio.sleep(0.033)  # Aim for ~30fps (1000ms / 30fps = 33.3ms per frame)
                            except ConnectionError:
                                logger.error(f"[WS] RTSP stream error for {rtsp_url}. Closing WebSocket stream for {session_id}.")
                                break # Exit the streaming loop
                            except asyncio.CancelledError:
                                logger.info(f"[WS] Stream task for {session_id} cancelled for {rtsp_url}.")
                                break
                            except Exception as e:
                                logger.error(f"[WS] Error during WebSocket streaming for {rtsp_url}: {e}", exc_info=True)
                                await asyncio.sleep(0.1)

                    stream_task = asyncio.create_task(stream_video())
                    # Keep the websocket open until the client disconnects or an error occurs
                    await stream_task

                except ConnectionError as e:
                    logger.error(f"[WS] Could not initialize RTSP stream for WebSocket {rtsp_url}: {e}")
                    await ws.send_str(json.dumps({"error": f"Failed to open RTSP stream: {e}"}))
                except Exception as e:
                    logger.exception(f"[WS] Unexpected error setting up WebSocket stream for {rtsp_url}:")
                    await ws.send_str(json.dumps({"error": f"Internal server error: {e}"}))

            elif msg.type == WSMsgType.ERROR:
                logger.error(f'[WS] WebSocket Error for client {session_id}: {ws.exception()}')
                break # Exit the message loop on error
            elif msg.type == WSMsgType.CLOSE:
                logger.info(f"[WS] Client {session_id} sent close frame.")
                break # Exit the message loop

    except asyncio.CancelledError:
        logger.info(f"[WS] WebSocket handler for client {session_id} cancelled.")
    except Exception as e:
        logger.exception(f"[WS] Unhandled exception in WebSocket handler for client {session_id}:")
    finally:
        logger.info(f"[WS] Client disconnected: {session_id}")
        if stream_task and not stream_task.done():
            stream_task.cancel()
            await asyncio.gather(stream_task, return_exceptions=True) # Ensure task is cancelled
        if frame_reader:
            frame_reader.stop()
        if session_id in clients:
            del clients[session_id]
        if not ws.closed:
            await ws.close()
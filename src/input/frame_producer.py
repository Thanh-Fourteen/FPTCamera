"""
FrameProducer continuously reads frames from an InputSource in a separate thread and pushes them into a FrameBuffer.
Handles reconnection logic for unstable sources (e.g., RTSP).
"""

import threading
import time
from typing import Any, Dict, Optional

from input.frame_buffer import FrameBuffer
from input.base import InputSource
from utils.logger import LoggerManager


class FrameProducer:
    """
    Producer that pulls frames from InputSource and pushes into FrameBuffer.
    """

    def __init__(
        self,
        source_path: str,
        source: InputSource,
        buffer: FrameBuffer,
        logger: LoggerManager,
        camera_id: str,
        area_name: str,
        reconnect_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.source_path = source_path
        self.source = source
        self.buffer = buffer
        self.logger = logger
        self.reconnect_config = reconnect_config or {"attempts": 3, "timeout_s": 5}

        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.frames_produced = 0
        self.camera_id = camera_id
        self.area_name = area_name

    def start(self) -> None:
        """Starts the producer thread and opens the source."""
        if not self.source.open(self.source_path):
            self.logger.log_error(
                self.camera_id, self.area_name, "Failed to open input source."
            )
        self.thread.start()

    def stop(self) -> None:
        """Signals the producer thread to stop and waits for it."""
        self._stop_event.set()
        self.thread.join()
        self.source.close()

    def _run(self) -> None:
        """Internal loop to read and buffer frames."""
        attempts = 0
        while not self._stop_event.is_set():
            if not self.source.is_opened():
                # Attempt reconnect
                if attempts >= self.reconnect_config["attempts"]:
                    self.logger.log_error(
                        self.camera_id,
                        self.area_name,
                        "Exceeded max reconnect attempts.",
                    )
                    break
                self.logger.log_warning(
                    "FrameProducer",
                    "",
                    f"Reconnecting in {self.reconnect_config['timeout_s']}s...",
                )
                time.sleep(self.reconnect_config["timeout_s"])
                if self.source.open(self.source_path):
                    self.logger.log_info(
                        self.camera_id, self.area_name, "Reconnected input source."
                    )
                    attempts = 0
                else:
                    attempts += 1
                    continue

            success, frame = self.source.read()
            if not success or frame is None or frame.size == 0:
                # Read failed; close and trigger reconnect
                self.logger.log_warning(
                    self.camera_id, self.area_name, "Read failed, closing source."
                )
                self.source.close()
                continue

            self.buffer.push(frame)
            self.frames_produced += 1

        # Cleanup
        self.logger.log_info(
            "FrameProducer",
            "",
            f"Stopping producer after {self.frames_produced} frames.",
        )
"""
Thread-safe frame buffer implementation with configurable strategies.
"""

import threading
import time
from collections import deque
from typing import Any, Deque, Tuple


class BufferStrategy:
    """
    Abstract base class for buffer strategies.
    """

    def push(self, buffer: Deque[Any], frame: Any) -> bool:
        """Push frame into buffer. Return True if frame was dropped."""
        raise NotImplementedError

    def get(self, buffer: Deque[Any]) -> Tuple[bool, Any]:
        """Get frame from buffer. Return (success_flag, frame or None)."""
        raise NotImplementedError


class LatestFrameStrategy(BufferStrategy):
    """Keep only the latest frame."""

    def push(self, buffer: Deque[Any], frame: Any) -> bool:
        buffer.clear()
        buffer.append(frame)
        return False  # never drop incoming

    def get(self, buffer: Deque[Any]) -> Tuple[bool, Any]:
        if buffer:
            return True, buffer.pop()
        return False, None


class FixedSizeQueueStrategy(BufferStrategy):
    """Fixed-size queue dropping oldest frames when full."""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size

    def push(self, buffer: Deque[Any], frame: Any) -> bool:
        if len(buffer) >= self.max_size:
            buffer.popleft()  # drop oldest
        buffer.append(frame)
        return False

    def get(self, buffer: Deque[Any]) -> Tuple[bool, Any]:
        if buffer:
            return True, buffer.popleft()
        return False, None


class AdaptiveBufferStrategy(BufferStrategy):
    """Adjust buffer size based on producer/consumer speed dynamically."""

    def __init__(self, min_size: int = 5, max_size: int = 50):
        self.min_size = min_size
        self.max_size = max_size
        self.current_size = min_size

    def push(self, buffer: Deque[Any], frame: Any) -> bool:
        dropped = False
        if len(buffer) >= self.current_size:
            buffer.popleft()
            dropped = True
        buffer.append(frame)
        # adjust current_size: expand if drops, shrink if underutilized
        if dropped and self.current_size < self.max_size:
            self.current_size += 1
        elif (
            not dropped
            and len(buffer) < self.current_size // 2
            and self.current_size > self.min_size
        ):
            self.current_size -= 1
        return dropped

    def get(self, buffer: Deque[Any]) -> Tuple[bool, Any]:
        if buffer:
            return True, buffer.popleft()
        return False, None


class FrameBuffer:
    """
    Thread-safe buffer that uses a BufferStrategy for frame management.
    """

    def __init__(self, strategy: BufferStrategy):
        self.strategy = strategy
        self.buffer: Deque[Any] = deque()
        self.lock = threading.Lock()
        # metrics
        self.frames_added = 0
        self.frames_dropped = 0
        self.total_push_time = 0.0

    def push(self, frame: Any) -> None:
        start = time.time()
        with self.lock:
            dropped = self.strategy.push(self.buffer, frame)
            if dropped:
                self.frames_dropped += 1
            else:
                self.frames_added += 1
        self.total_push_time += time.time() - start

    def get(self) -> Tuple[bool, Any]:
        with self.lock:
            return self.strategy.get(self.buffer)

    def size(self) -> int:
        with self.lock:
            return len(self.buffer)

    def metrics(self) -> dict:
        """Return buffer health metrics."""
        with self.lock:
            return {
                "added": self.frames_added,
                "dropped": self.frames_dropped,
                "current_size": len(self.buffer),
                "avg_push_time_ms": (
                    self.total_push_time
                    / max(self.frames_added + self.frames_dropped, 1)
                )
                * 1000,
            }
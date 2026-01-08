"""Base classes for ByteTrack object tracks.

Defines:
    TrackState: Enumeration of track lifecycle states.
    BaseTrack: Abstract base for individual tracklets.
"""

import numpy as np
from collections import OrderedDict


from components.tracking.ByteTrack.kalman_filter import KalmanFilter


class TrackState(object):
    """Track lifecycle states."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    """Abstract base class for tracklets with Kalman filter integration.

    Attributes:
        track_id (int): Unique ID for the track.
        state (TrackState): Current track state.
        is_activated (bool): Whether the track is active.
        history (OrderedDict): Past positions.
        score (float): Last associated detection score.
    """

    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # previous && current state of track
    prev_state: str = "unknown"

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_count():
        BaseTrack._count = 0
        return BaseTrack._count

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
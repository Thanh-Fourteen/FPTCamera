from typing import List, Optional, Tuple

import numpy as np

from components.tracking.ByteTrack.kalman_filter import KalmanFilter
from components.tracking.ByteTrack.base import BaseTrack, TrackState
from components.tracking.ByteTrack import matching



class STrack(BaseTrack):
    """A single object tracklet, which is a Kalman filter with a bounding box.

    This class represents a single tracked object, maintaining its state through
    a Kalman filter and associating it with detection bounding boxes.

    Attributes:
        shared_kalman (KalmanFilter): A Kalman filter instance shared among STrack
            objects for prediction. This is a class-level attribute.
        _tlwh (np.ndarray): The bounding box in ``[top_left_x, top_left_y, width, height]``
            format. This is the initial bounding box when the track is created.
        kalman_filter (Optional[KalmanFilter]): The Kalman filter instance used for this
            specific track. Initialized during activation.
        mean (Optional[np.ndarray]): The mean state vector from the Kalman filter.
        covariance (Optional[np.ndarray]): The covariance matrix from the Kalman filter.
        is_activated (bool): True if the track has been successfully associated with a
            high-confidence detection and activated.
        score (float): The confidence score of the detection associated with this track.
        tracklet_len (int): The number of consecutive frames this track has been updated.
        det_idx (Optional[int]): The original index of the detection that this track
            is currently associated with. Used to map tracks back to input detections.
    """

    shared_kalman: KalmanFilter = KalmanFilter()
    """:meta private:"""

    def __init__(self, tlwh: np.ndarray, score: float) -> None:
        """Initializes an STrack instance.

        Args:
            tlwh (np.ndarray): Bounding box in the format
                ``[top_left_x, top_left_y, width, height]``.
            score (float): Confidence score of the detection.
        """
        super().__init__()  # Initialize BaseTrack attributes
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter: Optional[KalmanFilter] = None
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.is_activated: bool = False
        self.score: float = score
        self.tracklet_len: int = 0
        self.det_idx: Optional[int] = None  # Instance-level initialization

    def predict(self) -> None:
        """Predicts the current location using the Kalman filter."""
        if self.kalman_filter is None or self.mean is None or self.covariance is None:
            return

        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0  # Reset velocity if not actively tracked
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks: List["STrack"]) -> None:
        """Predicts the current location for multiple tracks using Kalman filter.

        Args:
            stracks (List[STrack]): A list of STrack objects to be predicted.
        """
        if not stracks:
            return

        means_to_predict = []
        covs_to_predict = []
        tracks_being_predicted = []

        for st in stracks:
            if st.mean is not None and st.covariance is not None:
                mean_copy = st.mean.copy()  # Safe: st.mean is not None here
                if st.state != TrackState.Tracked:
                    mean_copy[7] = 0  # Reset velocity
                means_to_predict.append(mean_copy)
                covs_to_predict.append(st.covariance)
                tracks_being_predicted.append(st)

        if not tracks_being_predicted:
            return

        multi_mean_np = np.asarray(means_to_predict)
        multi_covariance_np = np.asarray(covs_to_predict)

        if multi_mean_np.size == 0 or multi_covariance_np.size == 0:
            return

        predicted_means, predicted_covariances = STrack.shared_kalman.multi_predict(
            multi_mean_np, multi_covariance_np
        )

        for i, st_updated in enumerate(tracks_being_predicted):
            st_updated.mean = predicted_means[i]
            st_updated.covariance = predicted_covariances[i]

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        """Starts a new tracklet.

        Args:
            kalman_filter (KalmanFilter): The Kalman filter to be used for tracking.
            frame_id (int): The frame ID of the current frame.
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        if self.kalman_filter is None:
            return
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(
        self, new_track: "STrack", frame_id: int, new_id: bool = False
    ) -> None:
        """Re-activates a lost tracklet with a new detection.

        Args:
            new_track (STrack): The new detection (as an STrack) to re-activate this track.
            frame_id (int): The frame ID of the current frame.
            new_id (bool, optional): Whether to assign a new ID to the tracklet.
                Defaults to False.
        """
        if self.kalman_filter is None or self.mean is None or self.covariance is None:
            return

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.det_idx = new_track.det_idx  # Propagate det_idx

    def update(self, new_track: "STrack", frame_id: int) -> None:
        """Updates a matched track with a new detection.

        Args:
            new_track (STrack): The new detection (as an STrack) to update this track.
            frame_id (int): The frame ID of the current frame.
        """
        if self.kalman_filter is None or self.mean is None or self.covariance is None:
            return

        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.det_idx = new_track.det_idx  # Propagate det_idx

    @property
    def tlwh(self) -> np.ndarray:
        """Get current position in bounding box format ``[tl_x, tl_y, w, h]``.

        Returns:
            np.ndarray: The current bounding box.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """Convert bounding box to format ``[min_x, min_y, max_x, max_y]``.

        Returns:
            np.ndarray: The bounding box in tlbr format.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from ``[tl_x, tl_y, w, h]`` to ``[cx, cy, a, h]``.

        Args:
            tlwh (np.ndarray): Bounding box in ``[tl_x, tl_y, w, h]`` format.

        Returns:
            np.ndarray: Bounding box in ``[center_x, center_y, aspect_ratio, height]`` format.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self) -> np.ndarray:
        """Convert current bounding box to ``[cx, cy, a, h]`` format.

        Returns:
            np.ndarray: The current bounding box in xyah format.
        """
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        """Convert bounding box from ``[min_x, min_y, max_x, max_y]`` to ``[tl_x, tl_y, w, h]``.

        Args:
            tlbr (np.ndarray): Bounding box in ``[min_x, min_y, max_x, max_y]`` format.

        Returns:
            np.ndarray: Bounding box in ``[tl_x, tl_y, w, h]`` format.
        """
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from ``[tl_x, tl_y, w, h]`` to ``[min_x, min_y, max_x, max_y]``.

        Args:
            tlwh (np.ndarray): Bounding box in ``[tl_x, tl_y, w, h]`` format.

        Returns:
            np.ndarray: Bounding box in ``[min_x, min_y, max_x, max_y]`` format.
        """
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self) -> str:
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    """BYTETracker for multi-object tracking.

    Implements the BYTE tracking algorithm which associates detections with existing
    tracks in two stages: first with high-confidence detections, then with
    low-confidence detections to recover occluded objects.

    Attributes:
        tracked_stracks (List[STrack]): List of currently tracked STrack objects.
        lost_stracks (List[STrack]): List of STrack objects that were tracked but are
            currently lost (not associated with a detection in recent frames).
        removed_stracks (List[STrack]): List of STrack objects that have been removed
            (lost for too long).
        frame_id (int): Current frame number being processed.
        track_thresh (float): Confidence threshold for considering a detection as high-confidence.
        match_thresh (float): IoU matching threshold for associating detections to tracks.
        track_buffer (int): Number of frames to keep a track in the lost state before removing it.
        kalman_filter (KalmanFilter): Kalman filter instance used for track prediction and update.
        det_thresh (float): Detection threshold for initializing new tracks.
        buffer_size (int): Effective buffer size in frames, calculated from `frame_rate` and `track_buffer`.
        max_time_lost (int): Maximum number of frames a track can be lost before being removed.
        use_iou_score_only (bool): If True, use only IoU for matching, otherwise fuse with detection score.
    """

    def __init__(
        self,
        frame_rate: int = 30,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        use_iou_score_only: bool = False,
    ) -> None:
        """Initializes the BYTETracker.

        Args:
            frame_rate (int, optional): Frame rate of the video. Defaults to 30.
                Used to calculate `buffer_size`.
            track_thresh (float, optional): Tracking threshold for high-confidence
                detections. Defaults to 0.5.
            match_thresh (float, optional): IoU matching threshold. Defaults to 0.8.
            track_buffer (int, optional): Number of frames to buffer a lost track.
                Defaults to 30.
            use_iou_score_only (bool, optional): Whether to use only IoU for matching
                or fuse with detection scores. Defaults to False.
        """
        BaseTrack.reset_count()
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.track_thresh: float = track_thresh
        self.use_iou_score_only: bool = use_iou_score_only
        self.match_thresh: float = match_thresh

        self.frame_id: int = 0
        _frame_rate = frame_rate if frame_rate > 0 else 30
        self.track_buffer: int = track_buffer
        self.det_thresh: float = track_thresh + 0.1

        self.kalman_filter = KalmanFilter()
        self.change_frame_rate(_frame_rate)

    def change_frame_rate(self, frame_rate: int) -> None:
        """Changes the frame rate and updates dependent parameters.

        Args:
            frame_rate (int): New frame rate to be set. Must be positive.
        """
        if frame_rate <= 0:
            print(
                f"Warning: Invalid frame_rate {frame_rate} provided. Using 30fps for buffer calculation."
            )
            _frame_rate = 30
        else:
            _frame_rate = frame_rate

        self.frame_rate = _frame_rate
        self.buffer_size: int = int(_frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost: int = self.buffer_size

    def update(self, output_results: np.ndarray) -> Tuple[List[STrack], List[int]]:
        """Updates the tracker with new detections.

        Args:
            output_results (np.ndarray): Detection results from the model.
                Expected shape is ``(N, 5)`` where columns are
                ``[x1, y1, x2, y2, score]`` or ``(N, 6)`` where columns are
                ``[x1, y1, x2, y2, score, class_id]`` (class_id is used to
                potentially adjust score).

        Returns:
            Tuple[List[STrack], List[int]]:
                A list of currently active STrack objects and a list of their
                corresponding original detection indices from `output_results`.
        """
        self.frame_id += 1
        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks_this_frame: List[STrack] = []
        removed_stracks_this_frame: List[STrack] = []

        if output_results is None or len(output_results) == 0:
            for track in self.tracked_stracks:
                track.mark_lost()
                lost_stracks_this_frame.append(track)

            self.tracked_stracks = [
                t for t in self.tracked_stracks if t.state == TrackState.Tracked
            ]
            self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
            self.lost_stracks.extend(lost_stracks_this_frame)

            current_lost_tracks = []
            for track in self.lost_stracks:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks_this_frame.append(track)
                else:
                    current_lost_tracks.append(track)
            self.lost_stracks = current_lost_tracks
            self.removed_stracks.extend(removed_stracks_this_frame)
            self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
                self.tracked_stracks, self.lost_stracks
            )

            output_stracks = [
                track for track in self.tracked_stracks if track.is_activated
            ]
            out_indices = [getattr(t, "det_idx", -1) for t in output_stracks]
            return output_stracks, out_indices

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        elif (
            output_results.shape[1] == 6
        ):  # Assuming 6th col is a confidence/class score multiplier
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]
        else:
            raise ValueError(
                "output_results must have 5 or 6 columns, got "
                f"{output_results.shape[1]}"
            )

        remain_inds = scores >= self.track_thresh
        inds_second = (scores > 0.1) & (scores < self.track_thresh)

        original_indices = np.arange(len(scores))
        dets_high_score_orig_idx = original_indices[remain_inds]
        dets_low_score_orig_idx = original_indices[inds_second]

        dets_high_score = bboxes[remain_inds]
        scores_high_score = scores[remain_inds]
        dets_low_score = bboxes[inds_second]
        scores_low_score = scores[inds_second]

        detections_high: List[STrack] = []
        if len(dets_high_score) > 0:
            for i, (tlbr, s) in enumerate(zip(dets_high_score, scores_high_score)):
                track = STrack(STrack.tlbr_to_tlwh(tlbr), s)
                track.det_idx = dets_high_score_orig_idx[i]
                detections_high.append(track)

        detections_low: List[STrack] = []
        if len(dets_low_score) > 0:
            for i, (tlbr, s) in enumerate(zip(dets_low_score, scores_low_score)):
                track = STrack(STrack.tlbr_to_tlwh(tlbr), s)
                track.det_idx = dets_low_score_orig_idx[i]
                detections_low.append(track)

        unconfirmed_tracks = [t for t in self.tracked_stracks if not t.is_activated]
        tracked_active_tracks = [t for t in self.tracked_stracks if t.is_activated]

        strack_pool = joint_stracks(tracked_active_tracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        dists = matching.iou_distance(strack_pool, detections_high)
        if not self.use_iou_score_only and len(detections_high) > 0:
            dists = matching.fuse_score(dists, detections_high)

        matches, u_track_pool, u_detections_high = matching.linear_assignment(
            dists, thresh=self.match_thresh
        )

        for i_track_pool, i_det_high in matches:
            track = strack_pool[i_track_pool]
            det = detections_high[i_det_high]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)  # det_idx is propagated inside update
                activated_stracks.append(track)
            else:
                track.re_activate(
                    det, self.frame_id, new_id=False
                )  # det_idx is propagated inside re_activate
                refind_stracks.append(track)

        r_tracked_active_tracks = [
            strack_pool[i]
            for i in u_track_pool
            if strack_pool[i].state == TrackState.Tracked
        ]

        dists_low = matching.iou_distance(r_tracked_active_tracks, detections_low)
        matches_low, u_r_tracked_active, u_detections_low = matching.linear_assignment(
            dists_low,
            thresh=0.5,  # Using a different threshold for low-score matches
        )

        for i_r_track, i_det_low in matches_low:
            track = r_tracked_active_tracks[i_r_track]
            det = detections_low[i_det_low]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:  # Should not happen if r_tracked_active_tracks only contains TrackedState
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for i_r_track in u_r_tracked_active:
            track = r_tracked_active_tracks[i_r_track]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks_this_frame.append(track)

        current_u_detections_high = [detections_high[i] for i in u_detections_high]
        dists_unconfirmed = matching.iou_distance(
            unconfirmed_tracks, current_u_detections_high
        )
        if not self.use_iou_score_only and len(current_u_detections_high) > 0:
            dists_unconfirmed = matching.fuse_score(
                dists_unconfirmed, current_u_detections_high
            )

        matches_unconfirmed, u_unconfirmed, u_u_detections_high = (
            matching.linear_assignment(
                dists_unconfirmed,
                thresh=0.7,  # Threshold for unconfirmed tracks
            )
        )

        for i_unconfirmed, i_u_det_high in matches_unconfirmed:
            track = unconfirmed_tracks[i_unconfirmed]
            det = current_u_detections_high[i_u_det_high]
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        for i_unconfirmed in u_unconfirmed:
            track = unconfirmed_tracks[i_unconfirmed]
            track.mark_removed()
            removed_stracks_this_frame.append(track)

        final_u_detections_high = [
            current_u_detections_high[i] for i in u_u_detections_high
        ]
        for det in final_u_detections_high:
            if det.score >= self.det_thresh:
                det.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(
                    det
                )  # det_idx is already set on STrack creation

        # Update state lists
        # Process tracks marked lost in this frame or previously and check max_time_lost
        current_lost_tracks_after_updates = []
        for track in self.lost_stracks:  # Iterate over existing lost tracks
            if self.frame_id - track.end_frame > self.max_time_lost:
                if track.state != TrackState.Removed:  # Avoid marking multiple times
                    track.mark_removed()
                    removed_stracks_this_frame.append(track)
            else:
                current_lost_tracks_after_updates.append(track)
        self.lost_stracks = current_lost_tracks_after_updates

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks_this_frame)  # Add newly lost tracks
        self.lost_stracks = sub_stracks(self.lost_stracks, removed_stracks_this_frame)
        # Ensure lost_stracks are unique by ID, in case of complex transitions
        self.lost_stracks = list(
            {track.track_id: track for track in self.lost_stracks}.values()
        )

        self.removed_stracks.extend(removed_stracks_this_frame)
        # Ensure removed_stracks are unique by ID
        self.removed_stracks = list(
            {track.track_id: track for track in self.removed_stracks}.values()
        )

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        out_indices = [getattr(t, "det_idx", -1) for t in output_stracks]

        return output_stracks, out_indices


def joint_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
    """Joins two lists of STrack objects, ensuring uniqueness by track_id.

    Args:
        tlista (List[STrack]): The first list of tracks.
        tlistb (List[STrack]): The second list of tracks.

    Returns:
        List[STrack]: A new list containing unique tracks from both lists.
            Tracks from `tlista` are prioritized if duplicates exist.
    """
    exists = {t.track_id: t for t in tlista}
    for t in tlistb:
        if t.track_id not in exists:
            exists[t.track_id] = t
    return list(exists.values())


def sub_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
    """Subtracts tracks in `tlistb` from `tlista` based on track_id.

    Args:
        tlista (List[STrack]): The list of tracks to subtract from.
        tlistb (List[STrack]): The list of tracks to subtract.

    Returns:
        List[STrack]: A new list containing tracks from `tlista` that are not in `tlistb`.
    """
    track_ids_b = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in track_ids_b]


def remove_duplicate_stracks(
    stracksa: List[STrack], stracksb: List[STrack]
) -> Tuple[List[STrack], List[STrack]]:
    """Removes duplicate STrack objects between two lists.

    Prioritizes tracks in `stracksa`. If a track ID from `stracksa`
    is found in `stracksb`, it's removed from `stracksb`. This is to ensure
    a track isn't simultaneously in, for example, tracked and lost lists.

    Args:
        stracksa (List[STrack]): The primary list of tracks (e.g., active tracks).
        stracksb (List[STrack]): The secondary list of tracks (e.g., lost tracks).

    Returns:
        Tuple[List[STrack], List[STrack]]:
            The updated `stracksa` (unchanged by this function) and `stracksb`
            (with duplicates relative to `stracksa` removed).
    """
    ids_a = {track.track_id for track in stracksa}
    unique_stracksb = [track for track in stracksb if track.track_id not in ids_a]
    return stracksa, unique_stracksb
"""Matching utilities for ByteTrack algorithm.

Provides functions for cost assignment and IoU-based distance metrics:
- linear_assignment
- bbox_iou, bbox_ious
- iou_distance, fuse_score
"""

from typing import List, Tuple

import lap
import numpy as np


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve linear assignment problem in a cost matrix.

    Args:
        cost_matrix (np.ndarray): Cost matrix.
        thresh (float): Threshold for the cost matrix.

    Returns:
        matches (np.ndarray): Matches found in the cost matrix.
        unmatched_a (np.ndarray): Unmatched elements in the first set.
        unmatched_b (np.ndarray): Unmatched elements in the second set.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1]),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def bbox_iou(atlbr: np.ndarray, btlbr: np.ndarray) -> float:
    """
    Compute IoU between two bounding boxes.

    Args:
        atlbr (np.ndarray): First bounding box in the format [x1, y1, x2, y2].
        btlbr (np.ndarray): Second bounding box in the format [x1, y1, x2, y2].

    Returns:
        float: IoU value between the two bounding boxes.
    """
    # Calculate areas of overlap
    inter_xmin = max(atlbr[0], btlbr[0])
    inter_ymin = max(atlbr[1], btlbr[1])
    inter_xmax = min(atlbr[2], btlbr[2])
    inter_ymax = min(atlbr[3], btlbr[3])
    # If there is no overlap
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    # Calculate area of overlap and area of each bounding box
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = (atlbr[2] - atlbr[0]) * (atlbr[3] - atlbr[1])
    area2 = (btlbr[2] - btlbr[0]) * (btlbr[3] - btlbr[1])
    # Calculate IoU
    iou = inter_area / float(area1 + area2 - inter_area)
    return iou


def bbox_ious(atlbrs: List[np.ndarray], btlbrs: List[np.ndarray]) -> np.ndarray:
    """
    Compute cost based on IoU

    Args:
        atlbrs (list[np.ndarray]): List of bounding boxes in the format [x1, y1, x2, y2].
        btlbrs (list[np.ndarray]): List of bounding boxes in the format [x1, y1, x2, y2].

    Returns:
        np.ndarray: Array of IoU values between the bounding boxes.
    """
    if not atlbrs or not btlbrs:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    return np.array(
        [[bbox_iou(atlbr, btlbr) for btlbr in btlbrs] for atlbr in atlbrs],
        dtype=np.float32,
    )


def iou_distance(atracks: List, btracks: List) -> np.ndarray:
    """
    Compute cost based on IoU

    Args:
        atracks (list[STrack]): List of STrack objects representing the first set of tracks.
        btracks (list[STrack]): List of STrack objects representing the second set of tracks.

    Returns:
        np.ndarray: Cost matrix based on IoU values.
    """
    atlbrs = [
        np.asarray(track) if isinstance(track, np.ndarray) else np.asarray(track.tlbr)
        for track in atracks
    ]
    btlbrs = [
        np.asarray(track) if isinstance(track, np.ndarray) else np.asarray(track.tlbr)
        for track in btracks
    ]
    _ious = bbox_ious(atlbrs, btlbrs)
    return 1 - _ious


def fuse_score(cost_matrix: np.ndarray, detections: List) -> np.ndarray:
    """
    Fuse the cost matrix with detection scores.

    Args:
        cost_matrix (np.ndarray): Cost matrix based on IoU values.
        detections (list[STrack]): List of STrack objects representing the detections.

    Returns:
        np.ndarray: Fused cost matrix based on IoU values and detection scores.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
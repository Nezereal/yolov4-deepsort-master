# iou_matching.py
# vim: expandtab:ts=4:sw=4
# This script provides utilities to compute the Intersection Over Union (IoU)
# and use it as a cost metric for object tracking.

from __future__ import absolute_import  # Ensures compatibility with Python 2.x (if needed).
import numpy as np  # Import NumPy for numerical computations.
from . import linear_assignment  # Import the linear assignment module for cost matrix handling.

def iou(bbox, candidates):
    """
    Computes the Intersection Over Union (IoU) between a given bounding box and 
    a set of candidate bounding boxes.

    IoU is a metric that measures the overlap between two bounding boxes, 
    returning a value between 0 (no overlap) and 1 (perfect overlap).

    Parameters
    ----------
    bbox : ndarray
        A bounding box in the format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        An array of IoU values between the `bbox` and each candidate bounding box.
    """
    # Calculate the top-left and bottom-right corners of the main bbox.
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    
    # Calculate the top-left and bottom-right corners of the candidate bboxes.
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    # Compute the intersection box's top-left and bottom-right corners.
    tl = np.c_[
        np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],  # Top-left x-coordinate.
        np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]   # Top-left y-coordinate.
    ]
    br = np.c_[
        np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],  # Bottom-right x-coordinate.
        np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]   # Bottom-right y-coordinate.
    ]

    # Calculate width and height of the intersection box, ensuring no negative values.
    wh = np.maximum(0., br - tl)

    # Compute the areas of intersection and the original bounding boxes.
    area_intersection = wh.prod(axis=1)  # Intersection area.
    area_bbox = bbox[2:].prod()          # Area of the main bounding box.
    area_candidates = candidates[:, 2:].prod(axis=1)  # Areas of candidate bounding boxes.

    # IoU is the ratio of the intersection area to the union area.
    return area_intersection / (area_bbox + area_candidates - area_intersection)

def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    Computes a cost matrix using IoU as the distance metric.

    Tracks and detections are matched by minimizing the cost (1 - IoU).
    A higher IoU means a closer match, so lower cost values indicate better matches.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of track objects, each representing a tracked object.
    detections : List[deep_sort.detection.Detection]
        A list of detection objects, each representing a newly detected object.
    track_indices : Optional[List[int]]
        A list of indices specifying which tracks to match. If None, use all tracks.
    detection_indices : Optional[List[int]]
        A list of indices specifying which detections to match. If None, use all detections.

    Returns
    -------
    ndarray
        A cost matrix of shape `(len(track_indices), len(detection_indices))`.
        Each entry (i, j) represents the cost `1 - IoU` between the i-th track and j-th detection.
    """
    # Default to all tracks and detections if specific indices are not provided.
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    # Initialize the cost matrix with zeros.
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

    # Iterate over each track index to compute the cost with all detections.
    for row, track_idx in enumerate(track_indices):
        # If the track hasn't been updated for more than one frame, assign infinite cost.
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        # Get the bounding box of the current track.
        bbox = tracks[track_idx].to_tlwh()

        # Collect bounding boxes of all selected detections.
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])

        # Compute the IoU-based cost for this track and all detections.
        cost_matrix[row, :] = 1. - iou(bbox, candidates)

    return cost_matrix

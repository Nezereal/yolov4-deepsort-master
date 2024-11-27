# detection.py
# vim: expandtab:ts=4:sw=4
# This script defines a Detection class used for representing bounding box detections in a single image.

import numpy as np  # Importing NumPy for numerical operations.

class Detection(object):
    """
    This class represents a bounding box detection in a single image. 

    It includes information such as the bounding box's position and size, 
    the confidence score of the detection, the object's class name, 
    and a feature vector describing the object.

    Parameters
    ----------
    tlwh : array_like
        The bounding box in the format `(top-left x, top-left y, width, height)`.
    confidence : float
        The confidence score of the detection (e.g., from a machine learning model).
    class_name : str
        The class label associated with the detected object (e.g., "car", "pedestrian").
    feature : array_like
        A feature vector (e.g., embedding) that describes the detected object.

    Attributes
    ----------
    tlwh : ndarray
        The bounding box in the format `(top-left x, top-left y, width, height)`.
    confidence : float
        The detection confidence score.
    class_name : str
        The class label of the detected object.
    feature : ndarray
        A NumPy array representing the feature vector of the detected object.
    """

    def __init__(self, tlwh, confidence, class_name, feature):
        """
        Initializes a Detection object.

        Parameters
        ----------
        tlwh : array_like
            The bounding box in the format `(x, y, w, h)`.
        confidence : float
            The confidence score for the detection.
        class_name : str
            The class name of the detected object.
        feature : array_like
            A feature vector for describing the detected object.
        """
        self.tlwh = np.asarray(tlwh, dtype=np.float)  # Convert bounding box to NumPy array of floats.
        self.confidence = float(confidence)  # Ensure confidence is a float value.
        self.class_name = class_name  # Store the class name as a string.
        self.feature = np.asarray(feature, dtype=np.float32)  # Convert feature vector to a NumPy array of 32-bit floats.

    def get_class(self):
        """
        Returns the class name of the detected object.

        Returns
        -------
        str
            The class name of the detected object.
        """
        return self.class_name

    def to_tlbr(self):
        """
        Converts the bounding box format from `(top-left x, top-left y, width, height)` (tlwh) 
        to `(top-left x, top-left y, bottom-right x, bottom-right y)` (tlbr).
        
        Returns
        -------
        ndarray
            A NumPy array with the bounding box in the `(tlbr)` format.
        """
        ret = self.tlwh.copy()  # Create a copy of the bounding box to avoid modifying the original.
        ret[2:] += ret[:2]  # Calculate bottom-right coordinates by adding width and height to the top-left coordinates.
        return ret

    def to_xyah(self):
        """
        Converts the bounding box format from `(top-left x, top-left y, width, height)` (tlwh) 
        to `(center x, center y, aspect ratio, height)` (xyah).

        The aspect ratio is defined as `width / height`.

        Returns
        -------
        ndarray
            A NumPy array with the bounding box in the `(xyah)` format.
        """
        ret = self.tlwh.copy()  # Create a copy of the bounding box to avoid modifying the original.
        ret[:2] += ret[2:] / 2  # Calculate the center by adding half the width and height to the top-left coordinates.
        ret[2] /= ret[3]  # Calculate the aspect ratio as `width / height`.
        return ret

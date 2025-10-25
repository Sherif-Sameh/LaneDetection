from functools import partial

import cv2
import numpy as np
from numpy.typing import NDArray

from lane_detection.detectors.base import LaneDetector

ROIType = tuple[
    tuple[int, int],  # top-left
    tuple[int, int],  # bottom-left
    tuple[int, int],  # bottom-right
    tuple[int, int],  # top-left
]

class ClassicalLaneDetector(LaneDetector):
    """Classical computer vision-based lane detector using OpenCV.
    
    Args:
        binary_thresh: Minimum intensity for binary thresholding filter.
        equalize_hist: Apply histogram equilization to image before edge and line detections.
        roi: Optional coordinates of region of interest to extract for line detection from image.
        canny_min_thresh: Canny minimum gradient threshold for hysteresis thresholding.
        canny_max_thresh: Canny maximum gradient threshold for hysteresis thresholding
        hough_rho_res: Hough line transform resolution for rho (r) parameter in pixels.
        hough_theta_res: Hough line transform resolution for theta parameter in radians.
        hough_min_thresh: Hough line transform minimum threshold for number of intersections
            (i.e. votes in thbinary_threshe rho-theta space) needed to "detect" a line.
        hough_min_len: Hough line transform minimum number of points needed to form a line. Lines
            with less than this number of points are disregarded.
        hough_max_gap: Hough line transform maximum gap between two points to be considered in the
            same line.
        angle_thresh: Maximum allowed deviation of lane lines from the vertical axis.
        store_intermed: Store intermediate images (binary and edge images).
    """

    def __init__(
        self,
        binary_thresh: int = 127,
        equalize_hist: bool = False,
        roi: ROIType | None = None,
        canny_min_thresh: float = 100,
        canny_max_thresh: float = 200,
        hough_rho_res: float = 1,
        hough_theta_res: float = np.pi / 180,
        hough_min_thresh: int = 50,
        hough_min_len: float = 50,
        hough_max_gap: float = 10,
        angle_thresh: float = np.pi / 4,
        store_intermed: bool = False,
    ):
        self.equalize_hist = equalize_hist
        self.roi = np.array(roi, dtype=np.int32) if roi is not None else roi
        self.angle_thresh = angle_thresh
        self.store_intermed = store_intermed
        
        # Create partial functions around needed OpenCV functions with fixed parameters
        self.thresh_fn = partial(
            cv2.threshold,
            thresh=binary_thresh,
            maxval=255,
            type=cv2.THRESH_BINARY,
        )
        self.canny_fn = partial(
            cv2.Canny,
            threshold1=canny_min_thresh,
            threshold2=canny_max_thresh,
        )
        self.hough_tf_fn = partial(
            cv2.HoughLinesP,
            rho=hough_rho_res,
            theta=hough_theta_res,
            threshold=hough_min_thresh,
            minLineLength=hough_min_len,
            maxLineGap=hough_max_gap,
        )

        # For storing intermediate outputs if needed
        self.binary = None
        self.edge = None
    
    @property
    def intermed_outs(self) -> tuple[NDArray[np.uint8] | None, NDArray[np.uint8] | None]:
        """Return the detector's intermediate binary and edge image outputs.
        
        Returns:
            tuple
            - binary: (N, H, W) batch of binary image outputs from last batch of input image.
            - edge: (N, H, W) batch of edge image outputs from last batch of input image.
        """
        return self.binary, self.edge
    
    def detect_lanes(self, images: NDArray) -> NDArray:
        """Extract all lanes in the input images and return their correponding lane images.
        
        Note: This detector can only classify pixels as lane (1) or background (0) not distinguish
            between different lane instances.

        Args:
            images: (N, H, W, C) batch of input images.
        
        Returns:
            (N, H, W) batch of lane images, where each pixel is labeled as either background (0)
            or belonging to a lane (>0).
        """
        assert images.ndim == 4, \
            f"Expected a 4-dimensional input, received {images.ndim} dimensions."
        out_shape = images.shape[:-1]
        binary = np.zeros(out_shape, dtype=np.uint8)
        edge = np.zeros(out_shape, dtype=np.uint8)
        out = np.zeros(out_shape, dtype=np.uint8)

        for i, image in enumerate(images):
            binary[i] = self._threshold_image(image)
            edge[i] = self.canny_fn(binary[i])
            lines = self.hough_tf_fn(edge[i])
            lines = self._filter_lines(lines)
            out[i] = self._get_lane_image(out_shape[1], out_shape[2], lines)
        
        if self.store_intermed:
            self.binary = binary
            self.edge = edge
        
        return out
    
    def _threshold_image(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Transform the input image to a binary image using adaptive thresholding.


        To transform the image the following steps are carried out in order:
            1) Convert image to grayscale.
            2) Equalize its histograms if required.
            3) Apply binary adaptive thresholding to the whole image or ROI if given.
        
        Args:
            image: (H, W, 3) input RGB image.
        
        Returns:
            (H, W) Binary representation of input image after conversion.
        """
        out = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.equalize_hist:
            out = cv2.equalizeHist(out)
        _, out = self.thresh_fn(out)
        if self.roi is not None:
            mask = np.zeros_like(out)
            cv2.fillPoly(mask, [self.roi], color=255)
            out = cv2.bitwise_and(out, mask)
        return out

    def _filter_lines(self, lines: NDArray[np.int32] | None) -> NDArray | None:
        """Filter detected lines according to their angles with respect the vertical axis.
        
        Args:
            lines: (M, 1, 4) output lines from probabilistic Hough line transform.
        
        Returns:
            (S, 1, 4) array of remaining valid lines after filtering. S could be zero.
        """
        if lines is None:
            return lines
        
        dx = lines[:, 0, 2] - lines[:, 0, 0]
        dy = lines[:, 0, 3] - lines[:, 0, 1]
        mask = np.abs(np.arctan2(dx, np.abs(dy))) < self.angle_thresh
        lines = lines[mask]
        return lines
    
    def _get_lane_image(
        self,
        height: int,
        width: int,
        lines: NDArray[np.int32] | None,
    ) -> NDArray[np.uint8]:
        """Draw detected lines from probabilistic Hough line transform onto a black background.
        
        Args:
            height: Image height in pixels.
            width: Image width in pixels.
            lines: (M, 1, 4) output lines from probabilistic Hough line transform.

        Returns:
            (H, W) grayscale image where lanes are labeled with an intensity of 1. 
        """
        out = np.zeros((height, width), dtype=np.uint8)
        if lines is None:
            return out
        
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Remove singelton dimension
            cv2.line(out, (x1, y1), (x2, y2), (1,), thickness=2)
        return out
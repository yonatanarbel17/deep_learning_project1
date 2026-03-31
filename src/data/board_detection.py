"""
Board Detection and Perspective Transformation Module.

This module handles:
1. Detecting and isolating the chessboard in images
2. Cropping to the board region
3. Applying perspective transformation to get top-down view
4. Converting angled photos to perfect square boards
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from PIL import Image
import cv2


def detect_board_corners_opencv(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect chessboard corners using OpenCV's findChessboardCorners.
    This is the most accurate method for standard chessboards.
    
    Args:
        image: Input image as numpy array (BGR or grayscale)
        
    Returns:
        4 corner points as (4, 2) array or None if not found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Try to find internal 7x7 corners (8x8 squares)
    ret, corners = cv2.findChessboardCorners(
        gray, 
        (7, 7),  # Internal corners (8x8 squares = 7x7 corners)
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if not ret or corners is None:
        return None
    
    # Refine corners for better accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Extract the 4 outer corners from the 7x7 grid
    # Top-left: first corner
    # Top-right: 7th corner (end of first row)
    # Bottom-right: last corner
    # Bottom-left: 7th from last (start of last row)
    h, w = gray.shape
    corners_2d = corners.reshape(-1, 2)
    
    # Find outer corners by finding min/max x+y and x-y
    # This handles rotated boards
    sums = corners_2d.sum(axis=1)
    diffs = corners_2d[:, 0] - corners_2d[:, 1]
    
    top_left_idx = np.argmin(sums)
    bottom_right_idx = np.argmax(sums)
    top_right_idx = np.argmax(diffs)
    bottom_left_idx = np.argmin(diffs)
    
    outer_corners = np.array([
        corners_2d[top_left_idx],
        corners_2d[top_right_idx],
        corners_2d[bottom_right_idx],
        corners_2d[bottom_left_idx]
    ], dtype=np.float32)
    
    return outer_corners


def detect_board_corners_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect board corners using contour detection method.
    Based on approach from: https://github.com/dimasikson/chess-board-detector
    
    Pipeline:
    1. Adaptive threshold (cv2.adaptiveThreshold)
    2. Find contours (cv2.findContours)
    3. Select contour with largest area (cv2.contourArea)
    4. Find corners of that contour (cv2.arcLength -> cv2.approxPolyDP)
    
    Args:
        image: Input image as numpy array (BGR or grayscale)
        
    Returns:
        4 corner points as (4, 2) array or None if not found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Step 1: Adaptive threshold (as per the GitHub project)
    # This is more robust than Canny for chessboards
    thresh = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV,  # Inverted to get dark board on light background
        11,  # Block size
        2    # C constant
    )
    
    # Optional: Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 2: Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Step 3: Select contour with largest area
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Filter: contour should be at least 10% of image area
    min_area = gray.shape[0] * gray.shape[1] * 0.1
    if area < min_area:
        return None
    
    # Step 4: Find corners using arcLength and approxPolyDP
    peri = cv2.arcLength(largest_contour, True)
    
    # Try different epsilon values to get 4 corners
    # Start with 0.02 (as in the project) and increase if needed
    for epsilon_factor in [0.02, 0.03, 0.05, 0.08, 0.1]:
        approx = cv2.approxPolyDP(largest_contour, epsilon_factor * peri, True)
        
        # If we have exactly 4 vertices, use it
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            
            # Order the corners: top-left, top-right, bottom-right, bottom-left
            # More robust method: sort by y-coordinate first, then by x-coordinate
            rect = np.zeros((4, 2), dtype=np.float32)
            
            # Sort corners by y-coordinate (top to bottom)
            y_sorted_indices = np.argsort(corners[:, 1])
            top_row_indices = y_sorted_indices[:2]  # Two corners with smallest y
            bottom_row_indices = y_sorted_indices[2:]  # Two corners with largest y
            
            # Sort top row by x-coordinate (left to right)
            top_row = corners[top_row_indices]
            top_row_sorted = top_row[np.argsort(top_row[:, 0])]
            rect[0] = top_row_sorted[0]  # top-left
            rect[1] = top_row_sorted[1]  # top-right
            
            # Sort bottom row by x-coordinate (left to right)
            bottom_row = corners[bottom_row_indices]
            bottom_row_sorted = bottom_row[np.argsort(bottom_row[:, 0])]
            rect[3] = bottom_row_sorted[0]  # bottom-left
            rect[2] = bottom_row_sorted[1]  # bottom-right
            
            return rect
        
        # If we have more than 4 vertices, try to get 4 by using minAreaRect
        elif len(approx) >= 4:
            # Use minimum area rectangle to get 4 corners
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.float32)
            
            # Order the corners: top-left, top-right, bottom-right, bottom-left
            # More robust method: sort by y-coordinate first, then by x-coordinate
            rect_ordered = np.zeros((4, 2), dtype=np.float32)
            
            # Sort corners by y-coordinate (top to bottom)
            y_sorted_indices = np.argsort(box[:, 1])
            top_row_indices = y_sorted_indices[:2]  # Two corners with smallest y
            bottom_row_indices = y_sorted_indices[2:]  # Two corners with largest y
            
            # Sort top row by x-coordinate (left to right)
            top_row = box[top_row_indices]
            top_row_sorted = top_row[np.argsort(top_row[:, 0])]
            rect_ordered[0] = top_row_sorted[0]  # top-left
            rect_ordered[1] = top_row_sorted[1]  # top-right
            
            # Sort bottom row by x-coordinate (left to right)
            bottom_row = box[bottom_row_indices]
            bottom_row_sorted = bottom_row[np.argsort(bottom_row[:, 0])]
            rect_ordered[3] = bottom_row_sorted[0]  # bottom-left
            rect_ordered[2] = bottom_row_sorted[1]  # bottom-right
            
            return rect_ordered
    
    return None


def detect_and_isolate_board(image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect the chessboard in the image and return the isolated board region with corners.
    
    This function:
    1. Detects the 4 corners of the chessboard
    2. Returns the corners and a bounding box for the board
    
    Args:
        image: Input image as numpy array (BGR format)
        
    Returns:
        Tuple of (corners, isolated_board_image) or None if not found
        - corners: (4, 2) array of corner coordinates
        - isolated_board: Cropped image containing just the board region
    """
    # Try OpenCV chessboard detection first (most accurate)
    corners = detect_board_corners_opencv(image)
    
    # Fallback to contour detection if OpenCV method fails
    if corners is None:
        corners = detect_board_corners_contour(image)
    
    if corners is None:
        return None
    
    # Calculate bounding box for the board
    x_min = int(np.min(corners[:, 0]))
    x_max = int(np.max(corners[:, 0]))
    y_min = int(np.min(corners[:, 1]))
    y_max = int(np.max(corners[:, 1]))
    
    # Add some padding (10% on each side)
    h, w = image.shape[:2]
    pad_x = int((x_max - x_min) * 0.1)
    pad_y = int((y_max - y_min) * 0.1)
    
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)
    
    # Crop to board region
    isolated_board = image[y_min:y_max, x_min:x_max]
    
    # Adjust corner coordinates relative to cropped region
    adjusted_corners = corners.copy()
    adjusted_corners[:, 0] -= x_min
    adjusted_corners[:, 1] -= y_min
    
    return adjusted_corners, isolated_board


def get_chess_board_upper_look(
    image: Image.Image,
    output_size: int = 512,
    use_perspective: bool = True
) -> Image.Image:
    """
    Transform a chessboard image to top-down view using perspective transformation.
    
    This function:
    1. Detects and isolates the board region
    2. Applies perspective transformation to get "upper look"
    3. Returns a perfectly square, top-down view of the board
    
    Args:
        image: Input PIL Image
        output_size: Size of the output square image
        use_perspective: If False, just resize (for backward compatibility)
        
    Returns:
        PIL Image of the board in top-down view (output_size x output_size)
    """
    if not use_perspective:
        # Fallback: just resize
        return image.resize((output_size, output_size), Image.BILINEAR)
    
    # Convert PIL to OpenCV format
    img_cv = np.array(image)
    if len(img_cv.shape) == 3:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    else:
        # If grayscale, convert to BGR
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    
    # Step 1: Detect and isolate the board
    result = detect_and_isolate_board(img_cv)
    
    if result is None:
        # If board detection fails, fallback to simple resize
        return image.resize((output_size, output_size), Image.BILINEAR)
    
    corners, isolated_board = result
    
    # Step 2: Apply perspective transformation on the isolated board
    # Define destination points for perspective transform (perfect square)
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, dst)
    
    # Apply perspective transformation
    warped = cv2.warpPerspective(isolated_board, M, (output_size, output_size))
    
    # Convert back to PIL Image
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(warped_rgb)

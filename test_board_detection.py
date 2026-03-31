#!/usr/bin/env python3
"""
Test script to apply board detection and perspective transformation on an image.
"""

import sys
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.board_detection import get_chess_board_upper_look, detect_and_isolate_board
import cv2
import numpy as np


def test_board_detection(image_path: str, output_path: str = "outputs/board_detection_test.png"):
    """
    Test board detection and perspective transformation.
    
    Args:
        image_path: Path to input image
        output_path: Where to save the result
    """
    print(f"Loading image from: {image_path}")
    img = Image.open(image_path).convert("RGB")
    print(f"Original image size: {img.size}")
    
    # Convert to OpenCV format for detection
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Step 1: Detect and isolate board
    print("\nStep 1: Detecting and isolating board...")
    result = detect_and_isolate_board(img_cv)
    
    if result is None:
        print("Warning: Could not detect board, using full image")
        isolated_board_cv = img_cv
        isolated_board_pil = img
    else:
        corners, isolated_board_cv = result
        isolated_board_pil = Image.fromarray(cv2.cvtColor(isolated_board_cv, cv2.COLOR_BGR2RGB))
        print(f"Isolated board size: {isolated_board_pil.size}")
        
        # Draw corners on isolated board for visualization
        isolated_with_corners = isolated_board_cv.copy()
        for i, corner in enumerate(corners):
            cv2.circle(isolated_with_corners, tuple(corners[i].astype(int)), 10, (0, 255, 0), -1)
            cv2.putText(isolated_with_corners, str(i), 
                       tuple(corners[i].astype(int) + [10, -10]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        isolated_with_corners_pil = Image.fromarray(cv2.cvtColor(isolated_with_corners, cv2.COLOR_BGR2RGB))
    
    # Step 2: Apply perspective transformation
    print("\nStep 2: Applying perspective transformation...")
    warped = get_chess_board_upper_look(
        img,
        output_size=800,
        use_perspective=True
    )
    print(f"Final warped image size: {warped.size}")
    
    # Create visualization with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title("1. Original Image", fontsize=14, weight='bold')
    axes[0].axis('off')
    
    # Isolated board
    if result is not None:
        axes[1].imshow(isolated_with_corners_pil)
        axes[1].set_title("2. Isolated Board (with detected corners)", fontsize=14, weight='bold')
    else:
        axes[1].imshow(img)
        axes[1].set_title("2. Board Detection Failed", fontsize=14, weight='bold', color='red')
    axes[1].axis('off')
    
    # Warped image
    axes[2].imshow(warped)
    axes[2].set_title("3. Top-Down View (Perspective Transform)", fontsize=14, weight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save result
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResult saved to: {output_path}")
    
    # Also save just the warped image
    warped_output = output_path.replace('.png', '_warped_only.png')
    warped.save(warped_output)
    print(f"Warped image saved to: {warped_output}")
    
    # Save isolated board
    if result is not None:
        isolated_output = output_path.replace('.png', '_isolated.png')
        isolated_board_pil.save(isolated_output)
        print(f"Isolated board saved to: {isolated_output}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test board detection on an image")
    parser.add_argument("--image", type=str, default="~/Downloads/tr.png",
                       help="Path to input image")
    parser.add_argument("--output", type=str, default="outputs/board_detection_test.png",
                       help="Output path for visualization")
    
    args = parser.parse_args()
    
    # Expand user path
    image_path = Path(args.image).expanduser()
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    test_board_detection(str(image_path), args.output)

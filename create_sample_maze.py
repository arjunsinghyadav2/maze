#!/usr/bin/env python3
"""
Sample Maze Generator

Creates a simple maze image with red and green circles for testing the maze solver.
"""

import cv2
import numpy as np


def create_sample_maze(width=640, height=480, output_path="sample_maze.png"):
    """
    Create a sample maze image with red and green circles.

    Args:
        width: Image width
        height: Image height
        output_path: Path to save the maze image
    """
    # Create a white image
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw maze walls (black lines)
    wall_color = (0, 0, 0)
    wall_thickness = 8

    # Outer border
    cv2.rectangle(image, (50, 50), (width - 50, height - 50), wall_color, wall_thickness)

    # Create a simple maze pattern
    # Horizontal walls
    cv2.line(image, (50, 150), (400, 150), wall_color, wall_thickness)
    cv2.line(image, (250, 250), (width - 50, 250), wall_color, wall_thickness)
    cv2.line(image, (50, 350), (400, 350), wall_color, wall_thickness)

    # Vertical walls
    cv2.line(image, (200, 50), (200, 120), wall_color, wall_thickness)
    cv2.line(image, (450, 180), (450, 320), wall_color, wall_thickness)
    cv2.line(image, (300, 280), (300, height - 50), wall_color, wall_thickness)

    # Draw red circle (start)
    red_center = (100, 100)
    cv2.circle(image, red_center, 20, (0, 0, 255), -1)
    cv2.circle(image, red_center, 22, (0, 0, 0), 2)  # Black outline

    # Draw green circle (end)
    green_center = (540, 400)
    cv2.circle(image, green_center, 20, (0, 255, 0), -1)
    cv2.circle(image, green_center, 22, (0, 0, 0), 2)  # Black outline

    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Sample maze created: {output_path}")
    print(f"Image size: {width}x{height} pixels")
    print(f"Red circle at: {red_center}")
    print(f"Green circle at: {green_center}")

    # Display the image
    cv2.imshow("Sample Maze", image)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image


if __name__ == "__main__":
    import sys

    output_path = "sample_maze.png"
    if len(sys.argv) > 1:
        output_path = sys.argv[1]

    create_sample_maze(output_path=output_path)

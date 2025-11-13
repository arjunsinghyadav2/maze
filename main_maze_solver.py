#!/usr/bin/env python3
"""
Main Maze Solver Script

This script:
1. Reads a maze image
2. Detects red circle and automatically finds the other circle
3. Allows user to specify which circle is start/end
4. Solves the maze using A* pathfinding
5. Converts pixel coordinates to robot coordinates
6. Moves the Dobot robot along the path
"""

import cv2
import numpy as np
import sys
import time
import pydobot

from maze_detector import detect_red_circle, detect_other_circle, preprocess_maze, visualize_detection
from maze_solver import a_star_search, simplify_path, visualize_path
from camera_utilities import apply_affine, apply_homography
from robot_utilities import move_to_home, move_to_specific_position, get_current_pose

# Pre-calibrated affine transformation matrix
M = np.array([
    [-7.03946756e-03, -4.69080162e-01, 3.69730111e+02],
    [-4.47296787e-01, 5.97834246e-03, 1.24360926e+02]
], dtype=np.float64)

# Pre-calibrated homography matrix (alternative)
H = np.array([
    [-2.44594058e-02, -4.75669460e-01, 3.67247188e+02],
    [-4.34041615e-01, 5.08065338e-03, 1.20901686e+02],
    [-5.98330506e-05, -7.62411614e-05, 1.00000000e+00]
], dtype=np.float64)


def move_robot_along_path(device, M, path, z_height=-45):
    """
    Move the robot along the given path.

    Args:
        device: Dobot device object
        M: Affine transformation matrix
        path: List of (x, y) pixel coordinates
        z_height: Z-coordinate for the robot (pen height)
    """
    print(f"\nMoving robot along path with {len(path)} waypoints...")

    for i, (u, v) in enumerate(path):
        # Convert pixel to robot coordinates
        Xa, Ya = apply_affine(M, u, v)
        print(f"[{i+1}/{len(path)}] Pixel({u}, {v}) -> Robot({Xa:.2f}, {Ya:.2f})")

        # Move to position
        move_to_specific_position(device, x=Xa, y=Ya, z=z_height)
        time.sleep(0.5)  # Small delay between moves

    print("\n✓ Done! Robot reached the end of the maze.")


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python main_maze_solver.py <maze_image_path> [--no-robot] [--step-size N] [--z-height Z]")
        print("\nOptions:")
        print("  --no-robot      : Run without connecting to robot (visualization only)")
        print("  --step-size N   : Simplify path by taking every Nth point (default: 5)")
        print("  --z-height Z    : Z-coordinate for robot movement (default: -45)")
        sys.exit(1)

    image_path = sys.argv[1]
    use_robot = "--no-robot" not in sys.argv

    # Parse step size
    step_size = 5
    if "--step-size" in sys.argv:
        idx = sys.argv.index("--step-size")
        if idx + 1 < len(sys.argv):
            step_size = int(sys.argv[idx + 1])

    # Parse z height
    z_height = -45
    if "--z-height" in sys.argv:
        idx = sys.argv.index("--z-height")
        if idx + 1 < len(sys.argv):
            z_height = float(sys.argv[idx + 1])

    # Load the maze image
    print(f"Loading maze image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)

    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

    # Detect red circle first
    print("\nDetecting circles...")
    red_pos = detect_red_circle(image)

    if red_pos is None:
        print("Error: Could not detect red circle!")
        sys.exit(1)

    print(f"Red circle found at: {red_pos}")

    # Detect the other circle (any color)
    other_pos = detect_other_circle(image, red_pos)

    if other_pos is None:
        print("Error: Could not detect the other circle!")
        print("Make sure there are two colored circles in the image.")
        sys.exit(1)

    print(f"Other circle found at: {other_pos}")

    # Visualize detected circles
    detection_vis = visualize_detection(image, red_pos, other_pos)
    cv2.imshow("Detected Circles", detection_vis)
    cv2.waitKey(1000)  # Show for 1 second

    # Ask user which circle is the start
    print("\n" + "="*50)
    print("Which circle should be the START?")
    print("  1. Red circle")
    print("  2. Other circle")
    print("="*50)

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == '1':
            start_pos = red_pos
            goal_pos = other_pos
            print("\n→ Start: RED circle, Goal: OTHER circle")
            break
        elif choice == '2':
            start_pos = other_pos
            goal_pos = red_pos
            print("\n→ Start: OTHER circle, Goal: RED circle")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    # Preprocess the maze
    print("\nPreprocessing maze...")
    binary_maze = preprocess_maze(image)
    cv2.imshow("Preprocessed Maze", binary_maze)
    cv2.waitKey(1000)

    # Solve the maze
    print("\nSolving maze with A* algorithm...")
    path = a_star_search(binary_maze, start_pos, goal_pos)

    if path is None:
        print("Error: No path found through the maze!")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit(1)

    print(f"✓ Path found! Length: {len(path)} pixels")

    # Simplify the path
    print(f"\nSimplifying path (taking every {step_size}th point)...")
    simplified_path = simplify_path(path, step_size=step_size)
    print(f"✓ Simplified path: {len(simplified_path)} waypoints")

    # Visualize the path
    path_vis = visualize_path(image, simplified_path, start_pos, goal_pos)
    cv2.imshow("Maze Solution", path_vis)

    # Save visualization
    output_path = image_path.replace('.', '_solution.')
    cv2.imwrite(output_path, path_vis)
    print(f"\n✓ Solution visualization saved to: {output_path}")

    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Move robot if requested
    if use_robot:
        print("\n" + "="*50)
        print("ROBOT CONTROL")
        print("="*50)
        proceed = input("Proceed with robot movement? (yes/no): ").strip().lower()

        if proceed == 'yes' or proceed == 'y':
            try:
                # Connect to Dobot
                print("\nConnecting to Dobot on /dev/ttyACM0...")
                device = pydobot.Dobot(port="/dev/ttyACM0")
                device.speed(50, 50)

                # Home the robot
                move_to_home(device)
                time.sleep(2)

                # Move along the path
                move_robot_along_path(device, M, simplified_path, z_height=z_height)

                # Close connection
                print("\nClosing robot connection...")
                device.close()
                print("✓ Robot connection closed.")

            except Exception as e:
                print(f"\nError controlling robot: {e}")
                sys.exit(1)
        else:
            print("Robot movement cancelled by user.")
    else:
        print("\n(Robot control skipped - running in visualization mode)")

    print("\n" + "="*50)
    print("Maze solving complete!")
    print("="*50)


if __name__ == "__main__":
    main()

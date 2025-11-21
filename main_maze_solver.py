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
from llm_maze_solver import solve_maze_with_llm_tools
from camera_utilities import apply_affine, apply_homography
from robot_utilities import move_to_home, move_to_specific_position, get_current_pose

# Pre-calibrated affine transformation matrix
M = np.array([
    [-7.03946756e-03, -4.69080162e-01, 4.19730111e+02],
    [-4.47296787e-01, 5.97834246e-03, 1.27360926e+02]
], dtype=np.float64)

# Pre-calibrated homography matrix (alternative)
H = np.array([
    [-2.44594058e-02, -4.75669460e-01, 3.67247188e+02],
    [-4.34041615e-01, 5.08065338e-03, 1.20901686e+02],
    [-5.98330506e-05, -7.62411614e-05, 1.00000000e+00]
], dtype=np.float64)


def move_robot_along_path(device, M, path, z_height=-25):
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
    if len(sys.argv) < 1:
        print("Usage: python main_maze_solver.py [maze_image_path] [OPTIONS]")
        print("\nIf no image path provided, captures from camera (device 0)")
        print("\nOptions:")
        print("  --no-robot          : Run without connecting to robot (visualization only)")
        print("  --solver METHOD     : Pathfinding method: 'astar' or 'llm-tools' (default: astar)")
        print("  --step-size N       : Simplify path by taking every Nth point (default: 5)")
        print("  --z-height Z        : Z-coordinate for robot movement (default: -45)")
        print("  --wall-clearance N  : Safety margin around walls in pixels (default: 25)")
        print("  --goal-radius N     : A* goal radius - accept path within N pixels of goal (default: 5)")
        print("  --no-circle-removal : Skip circle removal and wall clearance (test raw edges)")
        print("  --debug             : Show detailed debug images at each processing step")
        print("\nLLM Tool Solver:")
        print("  Requires ANTHROPIC_API_KEY environment variable")
        print("  Uses Claude Sonnet 4.5 with code execution tools for iterative pathfinding")
        sys.exit(1)

    # Check if first argument is an image path or a flag
    image_path = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        image_path = sys.argv[1]

    use_robot = "--no-robot" not in sys.argv
    debug_mode = "--debug" in sys.argv
    no_circle_removal = "--no-circle-removal" in sys.argv

    # Parse solver method
    solver_method = "astar"  # default
    if "--solver" in sys.argv:
        idx = sys.argv.index("--solver")
        if idx + 1 < len(sys.argv):
            solver_method = sys.argv[idx + 1].lower()
            if solver_method not in ["astar", "llm-tools"]:
                print(f"Error: Invalid solver method '{solver_method}'. Use 'astar' or 'llm-tools'")
                sys.exit(1)

    # Parse step size
    step_size = 60
    if "--step-size" in sys.argv:
        idx = sys.argv.index("--step-size")
        if idx + 1 < len(sys.argv):
            step_size = int(sys.argv[idx + 1])

    # Parse z height
    z_height = -35
    if "--z-height" in sys.argv:
        idx = sys.argv.index("--z-height")
        if idx + 1 < len(sys.argv):
            z_height = float(sys.argv[idx + 1])

    # Parse wall clearance
    wall_clearance = 25
    if "--wall-clearance" in sys.argv:
        idx = sys.argv.index("--wall-clearance")
        if idx + 1 < len(sys.argv):
            wall_clearance = int(sys.argv[idx + 1])

    # Parse goal radius
    goal_radius = 5
    if "--goal-radius" in sys.argv:
        idx = sys.argv.index("--goal-radius")
        if idx + 1 < len(sys.argv):
            goal_radius = int(sys.argv[idx + 1])

    # Load or capture the maze image
    if image_path:
        print(f"Loading maze image: {image_path}")
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not load image from {image_path}")
            sys.exit(1)

        print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    else:
        # Capture from camera with same setup as open_camera.py
        print("No image path provided - capturing from camera (device 0)...")

        # Home the robot first (camera is mounted on end effector)
        if use_robot:
            print("\n" + "="*50)
            print("ROBOT HOMING (Camera Positioning)")
            print("="*50)
            print("Camera is mounted on robot end effector.")
            print("Moving robot to home position for consistent camera angle...")

            try:
                device = pydobot.Dobot(port="/dev/tty.usbmodem4796319814332")
                device.speed(50, 50)
                move_to_home(device)
                print("✓ Robot homed and ready for image capture")

                # Close connection - will reconnect later for maze solving
                device.close()
                print("✓ Robot connection closed")
                print("="*50 + "\n")
            except Exception as e:
                print(f"Error homing robot: {e}")
                print("Continuing with camera capture anyway...")
                print("WARNING: Camera position may not be at home!")
        else:
            print("\nWARNING: Running with --no-robot flag")
            print("Robot will NOT be moved to home position before capture")
            print("Camera position may not be consistent!")

        print("\nPosition your maze in front of the camera")
        print("Press SPACE to capture, ESC to exit")

        # Camera configuration (matches open_camera.py)
        cam_index = 0
        width, height = 640, 480

        cap = cv2.VideoCapture(cam_index, cv2.CAP_ANY)
        if not cap.isOpened():
            print(f"Error: Could not open camera device {cam_index}")
            sys.exit(1)

        # Set camera resolution and buffer
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        image = None
        while True:
            ret, frame = cap.read()
            if not ret:
                # Retry once on failure
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read from camera")
                    break

            # Display the frame
            cv2.imshow("Camera - Press SPACE to capture, ESC to exit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Capture cancelled")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == 32:  # SPACE
                image = frame.copy()
                print(f"Image captured: {image.shape[1]}x{image.shape[0]} pixels")
                break

        cap.release()
        cv2.destroyAllWindows()

        if image is None:
            print("Error: No image captured")
            sys.exit(1)

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
    print(f"\nPreprocessing maze (wall clearance: {wall_clearance}px)...")
    if no_circle_removal:
        print("Circle removal: SKIPPED (testing stage 8 raw edges)")
        print("Wall clearance: SKIPPED")
    else:
        print(f"Circle removal: Localized within 110px of start and goal positions")
    if debug_mode:
        print("Debug mode: Showing detailed preprocessing steps...")
    binary_maze = preprocess_maze(image, wall_clearance=wall_clearance, debug=debug_mode,
                                   start_pos=start_pos, goal_pos=goal_pos, circle_radius=110,
                                   skip_circle_removal=no_circle_removal)

    # Save the binary maze for debugging
    if image_path:
        binary_maze_path = image_path.replace('.', '_binary_maze.')
        binary_with_circles_path = image_path.replace('.', '_binary_with_circles.')
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        binary_maze_path = f"binary_maze_{timestamp}.png"
        binary_with_circles_path = f"binary_with_circles_{timestamp}.png"

    cv2.imwrite(binary_maze_path, binary_maze)
    print(f"✓ Binary maze saved to: {binary_maze_path}")

    # Create visualization with start and goal circles on binary image
    binary_vis = cv2.cvtColor(binary_maze, cv2.COLOR_GRAY2BGR)
    cv2.circle(binary_vis, start_pos, 20, (0, 0, 255), 3)  # Red circle at start
    cv2.putText(binary_vis, "START", (start_pos[0] - 30, start_pos[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.circle(binary_vis, goal_pos, 20, (0, 255, 0), 3)  # Green circle at goal
    cv2.putText(binary_vis, "GOAL", (goal_pos[0] - 25, goal_pos[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(binary_with_circles_path, binary_vis)
    print(f"✓ Binary maze with circles saved to: {binary_with_circles_path}")

    if not debug_mode:
        cv2.imshow("Preprocessed Maze", binary_maze)
        cv2.waitKey(1000)

    # Validate start and goal positions
    print("\nValidating start and goal positions...")
    h, w = binary_maze.shape
    print(f"  Maze size: {w}x{h}")
    print(f"  Start position: {start_pos} -> pixel value: {binary_maze[start_pos[1], start_pos[0]]}")
    print(f"  Goal position: {goal_pos} -> pixel value: {binary_maze[goal_pos[1], goal_pos[0]]}")

    if binary_maze[start_pos[1], start_pos[0]] == 0:
        print("  ⚠ WARNING: Start position is on a wall (black pixel)!")
    if binary_maze[goal_pos[1], goal_pos[0]] == 0:
        print("  ⚠ WARNING: Goal position is on a wall (black pixel)!")

    # Solve the maze using selected method
    if solver_method == "astar":
        # Use A* algorithm with goal radius
        print(f"\nSolving maze with A* algorithm (goal radius: {goal_radius}px)...")
        path = a_star_search(binary_maze, start_pos, goal_pos, verbose=True, goal_radius=goal_radius)
    elif solver_method == "llm-tools":
        # Use Claude Sonnet 4.5 LLM with code execution tools (iterative, reliable)
        print(f"\nSolving maze with Claude Sonnet 4.5 LLM (tool-based, iterative)...")
        path = solve_maze_with_llm_tools(binary_maze, start_pos, goal_pos, verbose=True)
    else:
        print(f"Error: Unknown solver method '{solver_method}'")
        sys.exit(1)

    if path is None:
        print("Error: No path found through the maze!")
        print("\nDebug info:")
        print(f"  Binary maze min/max: {binary_maze.min()}/{binary_maze.max()}")
        print(f"  Path pixels (255): {np.sum(binary_maze == 255)}")
        print(f"  Wall pixels (0): {np.sum(binary_maze == 0)}")
        print(f"  Wall percentage: {100 * np.sum(binary_maze == 0) / binary_maze.size:.1f}%")
        print(f"\nPlease check the saved binary maze: {binary_maze_path}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit(1)

    print(f"✓ Path found! Length: {len(path)} pixels")
    print(f"  Solver used: {solver_method.upper()}")

    # Simplify the path
    print(f"\nSimplifying path (taking every {step_size}th point)...")
    simplified_path = simplify_path(path, step_size=step_size)
    print(f"✓ Simplified path: {len(simplified_path)} waypoints")

    # Visualize the path
    path_vis = visualize_path(image, simplified_path, start_pos, goal_pos)
    cv2.imshow("Maze Solution", path_vis)

    # Save visualization
    if image_path:
        output_path = image_path.replace('.', '_solution.')
    else:
        # Generate filename with timestamp for camera captures
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"maze_solution_{timestamp}.png"

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
                print("\nConnecting to Dobot...")
                device = pydobot.Dobot(port="/dev/tty.usbmodem4796319814332")
                device.speed(50, 50)

                # Home the robot (ensures starting position before maze solving)
                print("Homing robot before starting maze path...")
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

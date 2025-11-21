#!/usr/bin/env python3
"""
Test script to verify the maze solver setup.
"""

import sys


def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")

    try:
        import cv2
        print("  ✓ opencv-python (cv2) imported successfully")
    except ImportError as e:
        print(f"  ✗ opencv-python import failed: {e}")
        return False

    try:
        import numpy as np
        print("  ✓ numpy imported successfully")
    except ImportError as e:
        print(f"  ✗ numpy import failed: {e}")
        return False

    try:
        import pydobot
        print("  ✓ pydobot imported successfully")
    except ImportError as e:
        print(f"  ✗ pydobot import failed: {e}")
        print("    Note: This is only needed for robot control")

    try:
        from maze_detector import detect_circles, preprocess_maze, visualize_detection
        print("  ✓ maze_detector module imported successfully")
    except ImportError as e:
        print(f"  ✗ maze_detector import failed: {e}")
        return False

    try:
        from maze_solver import a_star_search, simplify_path, visualize_path
        print("  ✓ maze_solver module imported successfully")
    except ImportError as e:
        print(f"  ✗ maze_solver import failed: {e}")
        return False

    try:
        from camera_utilities import apply_affine, apply_homography
        print("  ✓ camera_utilities module imported successfully")
    except ImportError as e:
        print(f"  ✗ camera_utilities import failed: {e}")
        return False

    try:
        from robot_utilities import move_to_home, move_to_specific_position
        print("  ✓ robot_utilities module imported successfully")
    except ImportError as e:
        print(f"  ✗ robot_utilities import failed: {e}")
        return False

    return True


def test_coordinate_transform():
    """Test coordinate transformation functions."""
    print("\nTesting coordinate transformations...")

    try:
        import numpy as np
        from camera_utilities import apply_affine

        # Test affine transformation
        M = np.array([
            [-7.03946756e-03, -4.69080162e-01, 3.69730111e+02],
            [-4.47296787e-01, 5.97834246e-03, 1.24360926e+02]
        ], dtype=np.float64)

        u, v = 288, 255
        x, y = apply_affine(M, u, v)
        print(f"  ✓ Affine transform: pixel({u}, {v}) -> robot({x:.2f}, {y:.2f})")

        return True
    except Exception as e:
        print(f"  ✗ Coordinate transform test failed: {e}")
        return False


def test_pathfinding():
    """Test A* pathfinding algorithm."""
    print("\nTesting A* pathfinding...")

    try:
        import numpy as np
        from maze_solver import a_star_search

        # Create a simple 10x10 maze
        maze = np.ones((10, 10), dtype=np.uint8) * 255  # All white (path)

        # Add some walls
        maze[5, 2:8] = 0  # Horizontal wall
        maze[2:5, 5] = 0  # Vertical wall

        # Test pathfinding
        start = (1, 1)
        goal = (8, 8)

        path = a_star_search(maze, start, goal)

        if path:
            print(f"  ✓ Path found from {start} to {goal} (length: {len(path)})")
            return True
        else:
            print(f"  ✗ No path found from {start} to {goal}")
            return False
    except Exception as e:
        print(f"  ✗ Pathfinding test failed: {e}")
        return False


def main():
    print("="*50)
    print("Maze Solver Setup Test")
    print("="*50)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False
        print("\n⚠ Some imports failed. Please install missing packages:")
        print("  pip install -r requirements.txt")

    # Test coordinate transformation
    if not test_coordinate_transform():
        all_passed = False

    # Test pathfinding
    if not test_pathfinding():
        all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("✓ All tests passed!")
        print("\nYou can now:")
        print("  1. Create a sample maze: python create_sample_maze.py")
        print("  2. Run the solver: python main_maze_solver.py sample_maze.png --no-robot")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

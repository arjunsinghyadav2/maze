# Maze Solver with Dobot Integration

A comprehensive maze solving system that:
1. Reads a maze image
2. Detects red circle and automatically finds the other circle (any color)
3. Solves the maze using A* pathfinding algorithm
4. Converts pixel coordinates to Dobot robot coordinates
5. Moves the Dobot robot along the solved path

## Files Structure

### Main Components
- **`main_maze_solver.py`** - Main script to run the complete maze solving pipeline
- **`maze_detector.py`** - Image processing module for detecting circles and preprocessing maze
- **`maze_solver.py`** - A* pathfinding algorithm implementation
- **`create_sample_maze.py`** - Utility to generate sample maze images for testing

### Supporting Modules
- **`camera_utilities.py`** - Coordinate transformation functions (affine/homography)
- **`robot_utilities.py`** - Dobot robot control functions
- **`Affine_transform.py`** - Calibration and transformation matrix generation
- **`get_pixel_cordinates.py`** - Interactive tool to get pixel coordinates from camera

## Requirements

```bash
pip install opencv-python numpy pydobot
```

## Quick Start

### 1. Generate a Sample Maze (for testing)

```bash
python create_sample_maze.py sample_maze.png
```

This creates a simple maze image with red and green circles.

### 2. Run the Maze Solver (Visualization Only)

```bash
python main_maze_solver.py sample_maze.png --no-robot
```

This will:
- Detect the red and green circles
- Ask which circle is the start
- Solve the maze
- Display the solution
- Save the visualization

### 3. Run with Dobot Robot

```bash
python main_maze_solver.py sample_maze.png
```

This will perform all steps above and then move the Dobot robot along the path.

## Usage Options

```bash
python main_maze_solver.py <maze_image_path> [OPTIONS]
```

### Options:
- `--no-robot` : Run without connecting to robot (visualization only)
- `--step-size N` : Simplify path by taking every Nth point (default: 5)
- `--z-height Z` : Z-coordinate for robot movement (default: -45)

### Examples:

```bash
# Solve maze with default settings
python main_maze_solver.py my_maze.png

# Visualization only (no robot)
python main_maze_solver.py my_maze.png --no-robot

# Use finer path resolution
python main_maze_solver.py my_maze.png --step-size 3

# Adjust pen height
python main_maze_solver.py my_maze.png --z-height -50
```

## How It Works

### 1. Circle Detection
The system detects circles using HSV color space filtering:
- **Red circle**: HSV range [0-10, 160-180] with saturation > 100
- **Other circle**: Automatically detected as any saturated colored region that isn't the red circle

### 2. Maze Preprocessing
- Converts image to grayscale
- Applies Gaussian blur to reduce noise
- Uses adaptive thresholding to extract maze structure
- Removes colored circle areas to avoid interference
- Result: Binary image (0 = wall, 255 = path)

### 3. Pathfinding (A* Algorithm)
- Uses A* search algorithm with Manhattan distance heuristic
- Finds optimal path from start to goal
- 4-connected grid (up, down, left, right movements)

### 4. Path Simplification
- Reduces number of waypoints by taking every Nth point
- Always includes start and end points
- Reduces robot movement time

### 5. Coordinate Transformation
- Converts pixel coordinates to robot workspace coordinates
- Uses pre-calibrated affine transformation matrix
- Formula: `[X, Y]ᵀ = M * [u, v, 1]ᵀ`

### 6. Robot Control
- Homes the robot to origin
- Moves through each waypoint sequentially
- Announces "Done!" when reaching the end

## Calibration

If you need to recalibrate the pixel-to-robot coordinate transformation:

1. Use `get_pixel_cordinates.py` to click on known points in the camera view
2. Move robot to corresponding positions and record coordinates
3. Update the calibration points in `Affine_transform.py`
4. Run the calibration to get new transformation matrix M
5. Update the matrix M in `main_maze_solver.py`

## Creating Your Own Maze

Your maze image should have:
1. **White background** for paths
2. **Black walls** for obstacles
3. **One red circle** as a marker (start or end)
4. **One other colored circle** (any color: green, blue, yellow, etc.) as the second marker

The system automatically detects the red circle and then finds any other colored circle. The user will be prompted to choose which circle is the start point.

## Troubleshooting

### "Could not detect red circle"
- Ensure the red circle is clearly visible in good lighting
- Check that the red color is saturated enough (not too pale/pink)
- Adjust HSV ranges in `maze_detector.py` if needed

### "Could not detect the other circle"
- Ensure the second circle has a saturated color (not gray/white/black)
- The circle should be at least 50 pixels away from the red circle
- Try using brighter, more saturated colors (green, blue, yellow work well)

### "No path found"
- Check that maze has a valid path between start and goal
- Ensure walls are thick enough (at least 5-8 pixels)
- Try preprocessing with `--debug` flag to see binary maze

### Robot connection issues
- Verify Dobot is connected to `/dev/ttyACM0`
- Check USB connection and permissions
- Try different port if needed (update in `main_maze_solver.py`)

### Robot movements are inaccurate
- Recalibrate the affine transformation matrix
- Ensure camera position hasn't changed since calibration
- Verify robot workspace limits

## Advanced Usage

### Using from Camera Instead of Image File

You can modify `main_maze_solver.py` to capture from a camera:

```python
# Instead of cv2.imread(image_path)
cap = cv2.VideoCapture(0)
ret, image = cap.read()
cap.release()
```

### Adjusting Pathfinding

To use 8-connected pathfinding (including diagonals), modify `get_neighbors()` in `maze_solver.py`:

```python
def get_neighbors(pos):
    x, y = pos
    return [
        (x, y-1), (x, y+1), (x-1, y), (x+1, y),  # 4-connected
        (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)  # diagonals
    ]
```

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!

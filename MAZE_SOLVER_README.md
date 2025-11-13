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
- `--wall-clearance N` : Safety margin around walls in pixels (default: 5)
- `--debug` : Show detailed debug images at each processing step

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

# Increase wall clearance for wider paths
python main_maze_solver.py my_maze.png --wall-clearance 8

# Decrease wall clearance for narrow mazes
python main_maze_solver.py my_maze.png --wall-clearance 3

# Debug mode - see maze processing steps
python main_maze_solver.py my_maze.png --no-robot --debug
```

## How It Works

### 1. Circle Detection
The system detects circles using HSV color space filtering:
- **Red circle**: HSV range [0-10, 160-180] with saturation > 100
- **Other circle**: Two-stage detection to avoid noise:
  1. First, searches for green circles (broad HSV range 30-90)
  2. If no green found, searches for any saturated color with stricter area filtering

### 2. Maze Preprocessing
- Converts image to grayscale
- Applies Gaussian blur to reduce noise
- Uses **multiple thresholding methods**:
  - Simple threshold at 127 (mid-gray)
  - Otsu's thresholding (auto-calculated)
  - Combines with OR (detects walls from either method)
- Auto-detects if inversion needed (checks center region)
- **Morphological operations to fix broken lines**:
  - MORPH_CLOSE (5x5, 2 iterations) connects wall gaps
  - Wall dilation (3x3) strengthens thin lines
- Removes colored circle areas to avoid interference
- **Adds wall clearance** by dilating walls to create a safety margin
- Result: Binary image (0 = wall, 255 = path) with solid walls

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
- **Use green for best results** - the algorithm prioritizes green circles
- Ensure the second circle has a saturated color (not gray/white/black)
- The circle should be at least 50 pixels away from the red circle
- Make the circle large enough (at least 100 pixels² area)
- If using a non-green color, make it even larger (300+ pixels²) to avoid being filtered as noise

### "No path found" or "Path goes through walls"
**First, diagnose the problem with debug mode:**
```bash
python main_maze_solver.py maze.png --no-robot --debug
```

This shows you 7 processing stages:
1. Original image
2. Grayscale conversion
3. Simple threshold (127)
4. Otsu threshold
5. Combined with gap closing (walls should be SOLID BLACK, paths WHITE)
6. Color mask (circles removed)
7. Final with clearance

**If walls aren't detected properly (stage 5):**
- Ensure maze has good contrast (dark walls, light paths or vice versa)
- Walls should be solid black or very dark
- Paths should be white or very light
- Avoid gradients or shadows

**If path is blocked by clearance (stage 7):**
- Try reducing wall clearance: `--wall-clearance 2` or `--wall-clearance 0`
- The default clearance (5px) might be too large for narrow passages

**If path goes through walls even after proper detection:**
- This indicates a bug - please check the preprocessed maze window
- The path should only follow white pixels in the binary maze

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

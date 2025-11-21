# Maze Solver with Dobot Robot

An automated maze solving system that captures maze images, detects start/goal positions, finds the optimal path, and controls a Dobot robot to trace the solution.

## Quick Start

```bash
# Install dependencies
pip install opencv-python numpy pydobot anthropic

# Solve maze from camera (homes robot first for consistent camera angle)
python main_maze_solver.py

# Solve from image file
python main_maze_solver.py maze.png

# Visualization only (no robot movement)
python main_maze_solver.py --no-robot
```

## Command Line Options

```bash
python main_maze_solver.py [image_path] [OPTIONS]
```

**If no image path provided**: Captures from camera (device 0)

**Options:**
- `--solver METHOD` : Choose solver: `astar` (fast, default) or `llm-tools` (LLM with code execution)
- `--no-robot` : Visualization only, no robot connection
- `--step-size N` : Path simplification interval (default: 60)
- `--z-height Z` : Robot pen height in mm (default: -45)
- `--wall-clearance N` : Safety margin around walls in pixels (default: 25)
- `--debug` : Show detailed image processing steps

**Examples:**
```bash
# Use A* solver (fast, deterministic)
python main_maze_solver.py --solver astar

# Use LLM solver (Claude writes and executes pathfinding code)
python main_maze_solver.py --solver llm-tools

# Adjust wall clearance for narrow mazes
python main_maze_solver.py --wall-clearance 15

# Debug mode to see processing steps
python main_maze_solver.py --debug --no-robot
```

## System Architecture

### Core Pipeline (`main_maze_solver.py`)

1. **Image Acquisition**
   - Load from file OR capture from camera
   - If camera: homes robot first (camera mounted on end effector)

2. **Circle Detection** (`maze_detector.py`)
   - Detects red circle (HSV color filtering)
   - Finds other colored circle (prioritizes green)
   - User selects which circle is start/goal

3. **Maze Preprocessing** (`maze_detector.py`)
   - Converts to grayscale and applies Gaussian blur
   - **Sobel edge detection**: Computes gradient magnitude to find edges
   - Creates binary maze: 0=wall (black), 255=path (white)
   - Removes circles from maze to avoid blocking paths
   - Adds safety clearance around walls

4. **Pathfinding**
   - **A* Solver** (`maze_solver.py`): Fast, deterministic
   - **LLM Tool Solver** (`llm_maze_solver.py`): Claude iteratively writes/executes code

5. **Path Simplification**
   - Takes every Nth waypoint to reduce robot movements
   - Always keeps start and end points

6. **Coordinate Transformation** (`camera_utilities.py`)
   - Converts pixel (x, y) to robot coordinates (X, Y)
   - Uses pre-calibrated affine matrix: `[X, Y]ᵀ = M × [u, v, 1]ᵀ`

7. **Robot Execution** (`robot_utilities.py`)
   - Homes robot to starting position
   - Moves through each waypoint sequentially
   - Closes connection when done

### Supporting Modules

- **`Affine_transform.py`** - Calibration matrix generation
- **`get_pixel_cordinates.py`** - Interactive coordinate collection tool
- **`open_camera.py`** - Standalone camera preview utility

## How It Works

### Sobel Edge Detection (Maze Preprocessing)

The system uses **Sobel edge detection** to extract maze walls:

1. **Compute gradients**: Sobel operators compute horizontal (Gx) and vertical (Gy) intensity gradients
   ```
   Magnitude = √(Gx² + Gy²)
   ```

2. **Create edge map**: High gradient areas indicate edges (wall boundaries)

3. **Morphological operations**:
   - **Closing** (3x3 kernel, 2 iterations): Connects broken edges
   - **Dilation** (3x3 kernel, 2 iterations): Thickens walls
   - **Inversion**: Converts edges to binary (walls=black, paths=white)

4. **Localized circle removal**: Only removes circles within 110px of start/goal positions

5. **Wall clearance**: Dilates walls by N pixels for robot safety margin

**Why Sobel?** Detects edges based on intensity changes, works well with mazes that have clear wall boundaries.

### A* Pathfinding Algorithm

**A*** is an informed search algorithm that finds the shortest path efficiently.

**How it works:**
1. Maintains a priority queue ordered by **f(n) = g(n) + h(n)**
   - **g(n)**: Cost from start to node n (actual distance traveled)
   - **h(n)**: Heuristic estimate from n to goal (Manhattan distance)

2. **Algorithm steps:**
   ```
   1. Add start to open set with f(start) = h(start)
   2. While open set not empty:
      a. Pop node with lowest f-score
      b. If node is goal → reconstruct path and return
      c. For each valid neighbor:
         - Calculate tentative g-score
         - If better than previous → update and add to open set
   3. If open set empty → no path exists
   ```

3. **Manhattan distance heuristic**: `h(n) = |x₁ - x₂| + |y₁ - y₂|`
   - Admissible (never overestimates)
   - Guarantees optimal path

4. **4-connected movement**: Only up, down, left, right (no diagonals)

5. **Goal radius**: If goal is blocked, finds path within 5px and draws straight line

**Why A*?** Optimal, fast, and efficient for grid-based pathfinding.

### LLM Tool-Based Solver

Unlike traditional algorithms, the LLM solver uses **Claude Sonnet 4.5 with code execution tools**:

1. **Setup**: Saves binary maze to temporary file
2. **Iteration loop** (max 10 iterations):
   - Claude writes Python pathfinding code
   - Code executes in subprocess (30s timeout)
   - Claude sees output/errors and can debug
   - Extracts path from successful execution
3. **Advantages**: Observable, debuggable, flexible algorithm choice
4. **Cost**: ~$0.05 per maze, 10-30 seconds

Requires `ANTHROPIC_API_KEY` environment variable.

## Maze Requirements

Your maze image should have:
- **White background** for walkable paths
- **Black walls** for obstacles
- **One red circle** marking start or goal
- **One other colored circle** (green recommended) for the other marker

**Note**: The system will ask which circle is the start.

## Calibration

To recalibrate pixel-to-robot coordinate transformation:

1. Run `get_pixel_cordinates.py` to collect pixel coordinates from camera
2. Move robot to corresponding positions and record coordinates
3. Update calibration pairs in `Affine_transform.py`
4. Run calibration to compute new matrix M
5. Update M in `main_maze_solver.py` (lines 27-29)

**Important**: Camera must be at home position during calibration and image capture!

## Troubleshooting

### "Could not detect circles"
- Ensure good lighting and color saturation
- Red circle HSV range: [0-10, 160-180]
- Use green for the second circle (best detection)

### "No path found"
- Run with `--debug` to see preprocessing steps
- Check that walls appear as BLACK and paths as WHITE in binary image
- Reduce `--wall-clearance` if narrow passages are blocked

### Robot movements inaccurate
- Recalibrate transformation matrix
- Ensure camera at home position during capture
- Verify robot workspace limits

### "Start/Goal position is on a wall"
- Increase wall clearance or adjust circle positions
- Check binary maze in debug mode

## Project Structure

```
maze/
├── main_maze_solver.py       # Main pipeline
├── maze_detector.py           # Circle detection, Sobel preprocessing
├── maze_solver.py             # A* pathfinding
├── llm_maze_solver.py         # LLM tool-based solver
├── camera_utilities.py        # Affine/homography transforms
├── robot_utilities.py         # Dobot control functions
├── Affine_transform.py        # Calibration matrix generation
└── get_pixel_cordinates.py    # Calibration data collection
```

## Camera & Robot Setup

**Important**: Camera is mounted on Dobot end effector!

- When capturing from camera, robot automatically homes first
- Ensures consistent camera angle for calibration matrix
- Robot closes connection after homing, reconnects for maze solving

This is critical for accurate coordinate transformation.

---

**License**: Educational and research use
**Requirements**: Python 3.7+, OpenCV, NumPy, pydobot, anthropic (for LLM solver)

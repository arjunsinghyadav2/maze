"""
LLM-based Maze Solver using Claude Sonnet 4.5

This module provides an alternative to A* pathfinding by asking Claude to
implement the A* algorithm on the given maze data.
"""

import anthropic
import cv2
import numpy as np
import os
import json
from typing import List, Tuple, Optional


def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode OpenCV image (numpy array) to base64 string.

    Args:
        image: OpenCV image (BGR or grayscale)

    Returns:
        Base64 encoded string of PNG image
    """
    # Encode image as PNG
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Failed to encode image to PNG")

    # Convert to base64
    png_bytes = buffer.tobytes()
    base64_string = base64.b64encode(png_bytes).decode('utf-8')

    return base64_string


def create_annotated_maze_image(binary_maze: np.ndarray, start_pos: Tuple[int, int],
                                  goal_pos: Tuple[int, int]) -> np.ndarray:
    """
    Create an annotated version of the binary maze with start and goal marked.

    Args:
        binary_maze: Binary maze (0=wall, 255=path)
        start_pos: (x, y) start position
        goal_pos: (x, y) goal position

    Returns:
        Annotated BGR image
    """
    # Convert grayscale to BGR for color annotations
    annotated = cv2.cvtColor(binary_maze, cv2.COLOR_GRAY2BGR)

    # Draw start position (green circle)
    cv2.circle(annotated, start_pos, 15, (0, 255, 0), 3)
    cv2.circle(annotated, start_pos, 5, (0, 255, 0), -1)
    cv2.putText(annotated, "START", (start_pos[0] - 35, start_pos[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw goal position (red circle)
    cv2.circle(annotated, goal_pos, 15, (0, 0, 255), 3)
    cv2.circle(annotated, goal_pos, 5, (0, 0, 255), -1)
    cv2.putText(annotated, "GOAL", (goal_pos[0] - 30, goal_pos[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Add coordinate grid for reference (every 50 pixels)
    height, width = binary_maze.shape
    for x in range(0, width, 50):
        cv2.line(annotated, (x, 0), (x, height), (128, 128, 128), 1)
        if x > 0:
            cv2.putText(annotated, str(x), (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    for y in range(0, height, 50):
        cv2.line(annotated, (0, y), (width, y), (128, 128, 128), 1)
        if y > 0:
            cv2.putText(annotated, str(y), (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

    return annotated


def maze_to_simple_format(binary_maze: np.ndarray, sample_rate: int = 1) -> List[List[int]]:
    """
    Convert binary maze to a simple 2D list format for the LLM.

    Args:
        binary_maze: Binary maze (0=wall, 255=path)
        sample_rate: Sample every Nth pixel to reduce size (default=1, no sampling)

    Returns:
        2D list where 0=wall, 1=path
    """
    # Sample the maze to reduce size if needed
    if sample_rate > 1:
        sampled = binary_maze[::sample_rate, ::sample_rate]
    else:
        sampled = binary_maze

    # Convert to 0/1 format (0=wall, 1=path)
    maze_grid = (sampled > 128).astype(int).tolist()
    return maze_grid


def create_astar_prompt(maze_grid: List[List[int]], start_pos: Tuple[int, int],
                        goal_pos: Tuple[int, int]) -> str:
    """
    Create a prompt asking Claude to implement A* algorithm on the maze.

    Args:
        maze_grid: 2D list where 0=wall, 1=path
        start_pos: (x, y) start position
        goal_pos: (x, y) goal position

    Returns:
        Prompt string
    """
    height = len(maze_grid)
    width = len(maze_grid[0]) if height > 0 else 0

    # Convert maze to compact string representation for the prompt
    maze_str = json.dumps(maze_grid)

    prompt = f"""You are given a binary maze represented as a 2D array where:
- 0 = WALL (cannot pass through)
- 1 = PATH (can pass through)

**MAZE DATA:**
- Size: {width} x {height} (width x height)
- Coordinate system: maze[y][x] where (0,0) is top-left
- Start position: ({start_pos[0]}, {start_pos[1]})
- Goal position: ({goal_pos[0]}, {goal_pos[1]})

**THE MAZE ARRAY:**
```json
{maze_str}
```

**YOUR TASK:**
Implement the A* pathfinding algorithm to find the shortest path from start to goal.

**A* ALGORITHM REQUIREMENTS:**
1. Use Manhattan or Euclidean distance as the heuristic
2. Only move to cells where maze[y][x] == 1 (path cells)
3. Use 8-directional movement (up, down, left, right, and diagonals)
4. The path must stay within bounds: 0 <= x < {width}, 0 <= y < {height}
5. Each step should move to an adjacent cell (8-connected neighbors)

**OUTPUT FORMAT:**
Return ONLY valid JSON with the path as an array of [x, y] coordinates:

```json
{{
  "path": [
    [{start_pos[0]}, {start_pos[1]}],
    [x2, y2],
    [x3, y3],
    ...
    [{goal_pos[0]}, {goal_pos[1]}]
  ],
  "algorithm": "Brief description of how A* found this path"
}}
```

**IMPLEMENTATION NOTES:**
- Start with the start position in the open set
- Use f(n) = g(n) + h(n) where g is cost from start, h is heuristic to goal
- Always expand the node with lowest f-score
- Track parent nodes to reconstruct the path
- Return the complete path from start to goal as coordinate pairs

Implement A* pathfinding now and return the shortest path through the maze."""

    return prompt


def parse_llm_path_response(response_text: str) -> Optional[List[Tuple[int, int]]]:
    """
    Parse the LLM response to extract the path coordinates.

    Args:
        response_text: Raw response from Claude API

    Returns:
        List of (x, y) tuples representing the path, or None if parsing fails
    """
    try:
        # Try to find JSON in the response
        # Sometimes the LLM might add text before/after the JSON
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            print("  ✗ No JSON found in LLM response")
            return None

        json_str = response_text[start_idx:end_idx]
        data = json.loads(json_str)

        if 'path' not in data:
            print("  ✗ No 'path' key found in JSON response")
            return None

        # Convert to list of tuples
        path = [tuple(point) for point in data['path']]

        # Show algorithm description if provided
        if 'algorithm' in data:
            print(f"\n  LLM Algorithm: {data['algorithm']}")
        elif 'reasoning' in data:
            print(f"\n  LLM Reasoning: {data['reasoning']}")

        return path

    except json.JSONDecodeError as e:
        print(f"  ✗ Failed to parse JSON: {e}")
        print(f"  Response text: {response_text[:200]}...")
        return None
    except Exception as e:
        print(f"  ✗ Error parsing path: {e}")
        return None


def validate_path(path: List[Tuple[int, int]], binary_maze: np.ndarray,
                  start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> bool:
    """
    Validate that the LLM-generated path is valid.

    Args:
        path: List of (x, y) waypoints
        binary_maze: Binary maze (0=wall, 255=path)
        start_pos: Expected start position
        goal_pos: Expected goal position

    Returns:
        True if path is valid, False otherwise
    """
    if not path or len(path) < 2:
        print("  ✗ Path is empty or too short")
        return False

    # Check start and goal
    if path[0] != start_pos:
        print(f"  ⚠ Warning: Path start {path[0]} doesn't match expected {start_pos}")

    if path[-1] != goal_pos:
        print(f"  ⚠ Warning: Path end {path[-1]} doesn't match expected {goal_pos}")

    # Check if points are within bounds and on valid paths
    height, width = binary_maze.shape
    invalid_points = []

    for i, (x, y) in enumerate(path):
        # Check bounds
        if x < 0 or x >= width or y < 0 or y >= height:
            invalid_points.append((i, (x, y), "out of bounds"))
            continue

        # Check if on path (white pixel)
        if binary_maze[y, x] < 128:  # Wall or near-wall
            invalid_points.append((i, (x, y), "on wall"))

    if invalid_points:
        print(f"  ⚠ Warning: {len(invalid_points)} invalid points found:")
        for idx, pos, reason in invalid_points[:5]:  # Show first 5
            print(f"    Point {idx}: {pos} - {reason}")
        if len(invalid_points) > 5:
            print(f"    ... and {len(invalid_points) - 5} more")

    # Consider path valid if at least 80% of points are valid
    validity_ratio = (len(path) - len(invalid_points)) / len(path)
    return validity_ratio >= 0.8


def solve_maze_with_llm(binary_maze: np.ndarray, start_pos: Tuple[int, int],
                         goal_pos: Tuple[int, int], api_key: Optional[str] = None,
                         verbose: bool = False) -> Optional[List[Tuple[int, int]]]:
    """
    Solve the maze using Claude Sonnet 4.5 LLM by asking it to implement A* algorithm.

    Args:
        binary_maze: Binary maze image (0=wall, 255=path)
        start_pos: (x, y) start position
        goal_pos: (x, y) goal position
        api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        verbose: If True, print detailed information

    Returns:
        List of (x, y) waypoints from start to goal, or None if failed
    """
    if verbose:
        print("\n" + "="*60)
        print("LLM-BASED A* PATHFINDING (Claude Sonnet 4.5)")
        print("="*60)

    # Get API key
    if api_key is None:
        api_key = os.environ.get('ANTHROPIC_API_KEY')

    if not api_key:
        print("  ✗ Error: ANTHROPIC_API_KEY not found in environment")
        print("  Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return None

    # Convert maze to simple 2D format
    if verbose:
        print("  Converting maze to array format...")
    maze_grid = maze_to_simple_format(binary_maze, sample_rate=1)

    # Create prompt asking for A* implementation
    prompt = create_astar_prompt(maze_grid, start_pos, goal_pos)

    # Call Claude API
    if verbose:
        print("  Calling Claude Sonnet 4.5 API to implement A*...")
        print(f"  Maze size: {binary_maze.shape[1]}x{binary_maze.shape[0]}")
        print(f"  Start: {start_pos}, Goal: {goal_pos}")

    try:
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        response_text = message.content[0].text

        if verbose:
            print(f"  ✓ Received response from Claude ({len(response_text)} chars)")

        # Parse the response
        path = parse_llm_path_response(response_text)

        if path is None:
            print("  ✗ Failed to parse path from LLM response")
            return None

        if verbose:
            print(f"  ✓ Parsed {len(path)} waypoints from LLM response")

        # Validate the path
        if not validate_path(path, binary_maze, start_pos, goal_pos):
            print("  ⚠ Warning: Path validation found issues, but proceeding anyway")

        if verbose:
            print(f"  ✓ LLM A* path generated successfully!")

        return path

    except anthropic.APIError as e:
        print(f"  ✗ Anthropic API error: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Error calling LLM: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test with a simple maze
    print("LLM Maze Solver Module")
    print("Use this module by importing solve_maze_with_llm()")

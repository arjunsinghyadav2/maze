"""
LLM-based Maze Solver using Claude Sonnet 4.5

This module provides an alternative to A* pathfinding by using Claude's
vision and reasoning capabilities to solve the maze.
"""

import anthropic
import cv2
import numpy as np
import base64
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


def create_maze_solving_prompt(start_pos: Tuple[int, int], goal_pos: Tuple[int, int],
                                maze_shape: Tuple[int, int]) -> str:
    """
    Create a detailed prompt for Claude to solve the maze.

    Args:
        start_pos: (x, y) start position
        goal_pos: (x, y) goal position
        maze_shape: (height, width) of the maze

    Returns:
        Prompt string
    """
    height, width = maze_shape

    prompt = f"""You are solving a maze puzzle. Study the image carefully to find a VALID path through the maze corridors.

## CRITICAL RULES - READ CAREFULLY:

**THIS IS A MAZE - YOU CANNOT GO IN A STRAIGHT LINE!**

The maze has:
- **BLACK pixels (0)** = SOLID WALLS - You CANNOT pass through these!
- **WHITE pixels (255)** = OPEN CORRIDORS - You MUST stay in these areas!

**YOU MUST FOLLOW THE WHITE CORRIDORS. DO NOT CUT THROUGH WALLS.**

## IMAGE DETAILS:
- Size: {width}x{height} pixels
- Coordinate system: (0,0) is TOP-LEFT, x goes RIGHT, y goes DOWN
- Wall thickness: approximately 3-5 pixels (stay in center of corridors)
- Grid lines shown every 50 pixels for reference

## START AND GOAL:
- **START (GREEN circle)**: pixel ({start_pos[0]}, {start_pos[1]})
- **GOAL (RED circle)**: pixel ({goal_pos[0]}, {goal_pos[1]})

## YOUR TASK:
Trace a path through the WHITE corridors from START to GOAL.

**STEP-BY-STEP APPROACH:**
1. **Look at the maze image** - identify the black walls and white corridors
2. **Find the START (green circle)** - this is where you begin
3. **Find the GOAL (red circle)** - this is your destination
4. **Trace the white corridors** from start to goal - like following a road on a map
5. **Avoid all black areas** - these are walls you cannot pass through
6. **Stay in the CENTER of white corridors** - don't hug the walls
7. **Place waypoints every 10-20 pixels** along the corridor path

## PATH REQUIREMENTS:
✓ First waypoint = START position [{start_pos[0]}, {start_pos[1]}]
✓ Last waypoint = GOAL position [{goal_pos[0]}, {goal_pos[1]}]
✓ ALL waypoints between must be in WHITE areas (never black!)
✓ Stay 3-5 pixels away from walls (in center of corridors)
✓ Follow the natural turns and curves of the maze corridors
✓ Waypoints spaced every 10-20 pixels for smooth movement
✓ Total waypoints should be roughly Manhattan distance / 15

## WHAT NOT TO DO:
✗ DO NOT draw a straight line from start to goal
✗ DO NOT cut through black wall areas
✗ DO NOT place waypoints on or near black pixels
✗ DO NOT take shortcuts through walls
✗ DO NOT skip sections of the corridor

## OUTPUT FORMAT:
Return ONLY valid JSON (no other text):

```json
{{
  "path": [
    [{start_pos[0]}, {start_pos[1]}],
    [x2, y2],
    [x3, y3],
    ...
    [{goal_pos[0]}, {goal_pos[1]}]
  ],
  "reasoning": "Brief description of the route you traced through the corridors"
}}
```

## EXAMPLE OF GOOD REASONING:
"Started at green circle, followed white corridor going right, turned down at intersection, continued through winding corridor, turned left at junction, followed straight corridor to red circle goal."

**Remember: This is a MAZE. You must navigate through the corridors, not cut through walls!**

Now carefully examine the maze image and trace a valid path through the WHITE corridors."""

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

        # Show reasoning if provided
        if 'reasoning' in data:
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
    Solve the maze using Claude Sonnet 4.5 LLM.

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
        print("LLM-BASED MAZE SOLVING (Claude Sonnet 4.5)")
        print("="*60)

    # Get API key
    if api_key is None:
        api_key = os.environ.get('ANTHROPIC_API_KEY')

    if not api_key:
        print("  ✗ Error: ANTHROPIC_API_KEY not found in environment")
        print("  Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return None

    # Create annotated maze image for better understanding
    annotated_maze = create_annotated_maze_image(binary_maze, start_pos, goal_pos)

    # Encode image to base64
    if verbose:
        print("  Encoding maze image...")
    image_base64 = encode_image_to_base64(annotated_maze)

    # Create prompt
    prompt = create_maze_solving_prompt(start_pos, goal_pos, binary_maze.shape)

    # Call Claude API
    if verbose:
        print("  Calling Claude Sonnet 4.5 API...")
        print(f"  Maze size: {binary_maze.shape[1]}x{binary_maze.shape[0]}")
        print(f"  Start: {start_pos}, Goal: {goal_pos}")

    try:
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
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
            print(f"  ✓ LLM path generated successfully!")

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

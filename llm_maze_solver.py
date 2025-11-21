"""
LLM-based Maze Solver (Code Simulation Mode)

This module replaces the standard A* execution by sending the maze image, 
coordinates, and the A* source code to Claude, asking it to simulate 
the execution and return the resulting path.
"""

import anthropic
import cv2
import numpy as np
import base64
import os
import json
import re
from typing import List, Tuple, Optional

ASTAR_SOURCE_CODE = r"""
def is_valid_position(maze, pos):
    x, y = pos
    h, w = maze.shape
    if x < 0 or x >= w or y < 0 or y >= h: return False
    return maze[y, x] > 128

def get_neighbors(pos):
    x, y = pos
    return [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def euclidean_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def a_star_search(maze, start, goal, goal_radius=100):
    # Priority queue: (f_score, counter, position)
    import heapq
    counter = 0
    open_set = [(0, counter, start)]
    counter += 1
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, goal)}
    open_set_hash = {start}

    while open_set:
        current_f, _, current = heapq.heappop(open_set)
        open_set_hash.discard(current)

        dist_to_goal = euclidean_distance(current, goal)
        
        # Check exact goal or radius
        if current == goal or dist_to_goal <= goal_radius:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            
            # Add straight line if stopped at radius
            if path[-1] != goal:
                last_point = path[-1]
                dx = goal[0] - last_point[0]
                dy = goal[1] - last_point[1]
                num_steps = max(abs(dx), abs(dy))
                if num_steps > 0:
                    for i in range(1, num_steps + 1):
                        t = i / num_steps
                        path.append((int(last_point[0] + dx * t), int(last_point[1] + dy * t)))
                if path[-1] != goal: path.append(goal)
            return path

        for neighbor in get_neighbors(current):
            if not is_valid_position(maze, neighbor): continue
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f = tentative_g_score + manhattan_distance(neighbor, goal)
                f_score[neighbor] = f
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1
                    open_set_hash.add(neighbor)
    return None
"""

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode OpenCV image (numpy array) to base64 string."""
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return base64.b64encode(buffer.tobytes()).decode('utf-8')

def construct_simulation_prompt(start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> str:
    """Constructs the prompt containing the code to be executed."""
    
    prompt = f"""You are a Python Code Interpreter. 
            I have provided an image representing a Binary Maze (Numpy Array representation where 0=Wall, 255=Path).
            I have provided a specific Python implementation of the A* Search Algorithm below.

            YOUR TASK:
            Simulate the execution of the provided code on the provided image using the coordinates below. 
            Do not write new code. Execute the provided code logic step-by-step internally and return the final output variable 'path'.

            INPUT DATA:
            - Start Coordinate (x, y): {start_pos}
            - Goal Coordinate (x, y): {goal_pos}
            - Image Data: See attached image (White pixels are valid paths, Black are walls).

            ALGORITHM TO EXECUTE:
            ```python
            {ASTAR_SOURCE_CODE}
            OUTPUT FORMAT: Return ONLY the raw JSON list of tuples. Do not include markdown
            formatting, explanations, or the code itself. Example format: [[10, 10], [10, 11], [11, 11]...] """
    return prompt

def extract_json_from_response(response_text: str) -> Optional[List[Tuple[int, int]]]:
    """Extracts and parses the list of tuples from the LLM response."""
    try:
        # Find array brackets (handling multiline responses)
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if not match:
            return None
            
        json_str = match.group(0)
        # Fix common LLM JSON mistakes (converting parens to brackets for valid JSON)
        json_str = json_str.replace('(', '[').replace(')', ']')
        
        path_data = json.loads(json_str)
        
        # Convert back to tuples
        path = [tuple(point) for point in path_data]
        return path
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return None

def solve_maze_with_llm(binary_maze: np.ndarray, start_pos: Tuple[int, int],
                         goal_pos: Tuple[int, int], verbose: bool = False, **kwargs) -> Optional[List[Tuple[int, int]]]:
    """
    Solves the maze by asking Claude to execute the A* code on the provided image.
    
    Args:
        binary_maze: Binary maze image (0=wall, 255=path)
        start_pos: (x, y) start
        goal_pos: (x, y) goal
        verbose: Print debug info
        **kwargs: Absorbs extra arguments (like goal_radius) used by the standard solver
    
    Returns:
        List of (x, y) waypoints
    """
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found.")
        return None

    if verbose:
        print(f"LLM Solver: Encoding image ({binary_maze.shape}) and preparing code simulation...")

    # 1. Encode the binary maze image
    image_base64 = encode_image_to_base64(binary_maze)

    # 2. Construct Prompt with Code
    prompt_text = construct_simulation_prompt(start_pos, goal_pos)

    # 3. Call Claude
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        if verbose:
            print("LLM Solver: Sending request to Claude (Code Simulation Mode)...")

        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",  # Claude Sonnet 4.5
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
                            "text": prompt_text
                        }
                    ],
                }
            ],
        )

        response_text = message.content[0].text
        
        if verbose:
            print("LLM Solver: Response received. Parsing...")

        # 4. Parse Result
        path = extract_json_from_response(response_text)

        if path and len(path) > 0:
            if verbose:
                print(f"LLM Solver: Successfully recovered path of length {len(path)}")
            return path
        else:
            print("LLM Solver: Failed to parse a valid path from response.")
            return None

    except Exception as e:
        print(f"LLM Solver Error: {e}")
        return None


def solve_maze_with_llm_tools(binary_maze: np.ndarray, start_pos: Tuple[int, int],
                                goal_pos: Tuple[int, int], verbose: bool = False,
                                max_iterations: int = 10) -> Optional[List[Tuple[int, int]]]:
    """
    Solves the maze by giving Claude access to Python code execution tools.
    The LLM can iteratively write, execute, and debug code to find the path.

    Args:
        binary_maze: Binary maze image (0=wall, 255=path)
        start_pos: (x, y) start
        goal_pos: (x, y) goal
        verbose: Print debug info
        max_iterations: Maximum number of tool use iterations

    Returns:
        List of (x, y) waypoints
    """
    import tempfile
    import subprocess
    import sys

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found.")
        return None

    if verbose:
        print(f"LLM Tool Solver: Preparing maze for code execution...")

    # Save maze to temporary file so LLM code can access it
    temp_dir = tempfile.mkdtemp()
    maze_path = os.path.join(temp_dir, "maze.npy")
    np.save(maze_path, binary_maze)

    if verbose:
        print(f"  Maze saved to: {maze_path}")
        print(f"  Maze shape: {binary_maze.shape}")
        print(f"  Start: {start_pos}, Goal: {goal_pos}")

    # Define the Python execution tool
    tools = [
        {
            "name": "execute_python",
            "description": "Execute Python code and return the output (stdout, stderr, and return value). Use this to implement and test your pathfinding algorithm.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. The code has access to numpy, cv2, and standard libraries."
                    }
                },
                "required": ["code"]
            }
        }
    ]

    # Initial system message
    system_prompt = f"""You are a maze-solving expert with access to Python code execution.

TASK: Find a path from START to GOAL through the maze.

MAZE DATA:
- Location: {maze_path}
- Load with: maze = np.load('{maze_path}')
- Shape: {binary_maze.shape} (height={binary_maze.shape[0]}, width={binary_maze.shape[1]})
- Format: Binary numpy array where 0=wall (black), 255=path (white)
- Start position (x, y): {start_pos}
- Goal position (x, y): {goal_pos}

IMPORTANT COORDINATE SYSTEM:
- maze[y, x] accesses pixel at position (x, y)
- x is horizontal (column), y is vertical (row)
- Valid x range: [0, {binary_maze.shape[1]-1}]
- Valid y range: [0, {binary_maze.shape[0]-1}]

YOUR APPROACH:
1. Use the execute_python tool to write and run pathfinding code
2. Implement A* or any suitable pathfinding algorithm
3. Test your code and debug if needed
4. Return the final path as a JSON list of [x, y] coordinates

OUTPUT FORMAT (when done):
Return ONLY a JSON list: [[x1, y1], [x2, y2], ..., [xn, yn]]

You can execute code multiple times to debug and refine your solution."""

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Conversation history
        messages = [
            {
                "role": "user",
                "content": "Please find a path through the maze from start to goal. Use the execute_python tool to implement and test your pathfinding algorithm."
            }
        ]

        iteration = 0
        final_path = None

        if verbose:
            print("\nLLM Tool Solver: Starting iterative solving process...\n")

        while iteration < max_iterations:
            iteration += 1

            if verbose:
                print(f"{'='*60}")
                print(f"Iteration {iteration}/{max_iterations}")
                print(f"{'='*60}")

            # Call Claude with tools
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                system=system_prompt,
                tools=tools,
                messages=messages
            )

            if verbose:
                print(f"Stop reason: {response.stop_reason}")

            # Check if Claude wants to use a tool
            if response.stop_reason == "tool_use":
                # Process tool calls
                tool_results = []

                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_use_id = content_block.id

                        if verbose:
                            print(f"\n→ Tool call: {tool_name}")
                            print(f"  Code length: {len(tool_input['code'])} characters")

                        if tool_name == "execute_python":
                            # Execute the Python code
                            code = tool_input["code"]

                            if verbose:
                                print(f"\n  Executing code:")
                                print("  " + "-"*58)
                                code_lines = code.split('\n')
                                for line in code_lines[:10]:  # Show first 10 lines
                                    print(f"  {line}")
                                if len(code_lines) > 10:
                                    remaining = len(code_lines) - 10
                                    print(f"  ... ({remaining} more lines)")
                                print("  " + "-"*58)

                            # Execute in subprocess for safety
                            try:
                                result = subprocess.run(
                                    [sys.executable, "-c", code],
                                    capture_output=True,
                                    text=True,
                                    timeout=30,
                                    cwd=temp_dir
                                )

                                output = result.stdout
                                error = result.stderr
                                return_code = result.returncode

                                if return_code == 0:
                                    tool_result = f"Success!\nOutput:\n{output}"
                                    if error:
                                        tool_result += f"\nWarnings:\n{error}"
                                else:
                                    tool_result = f"Error (exit code {return_code}):\n{error}\nOutput:\n{output}"

                                if verbose:
                                    print(f"\n  Result:")
                                    print("  " + "-"*58)
                                    result_lines = tool_result.split('\n')
                                    for line in result_lines[:15]:
                                        print(f"  {line}")
                                    if len(result_lines) > 15:
                                        remaining_result = len(result_lines) - 15
                                        print(f"  ... ({remaining_result} more lines)")
                                    print("  " + "-"*58)

                            except subprocess.TimeoutExpired:
                                tool_result = "Error: Code execution timed out (30s limit)"
                                if verbose:
                                    print(f"  {tool_result}")
                            except Exception as e:
                                tool_result = f"Error executing code: {str(e)}"
                                if verbose:
                                    print(f"  {tool_result}")

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": tool_result
                            })

                # Add assistant's response and tool results to conversation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            elif response.stop_reason == "end_turn":
                # Claude has finished - extract the path from response
                response_text = ""
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        response_text += content_block.text

                if verbose:
                    print(f"\nLLM response:")
                    print(f"  {response_text[:200]}...")

                # Try to extract path from response
                path = extract_json_from_response(response_text)

                if path and len(path) > 0:
                    final_path = path
                    if verbose:
                        print(f"\n✓ Successfully extracted path with {len(path)} waypoints")
                    break
                else:
                    # Ask for the path explicitly
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": "Please provide the final path as a JSON list of coordinates: [[x1, y1], [x2, y2], ...]"
                    })
            else:
                # Unexpected stop reason
                if verbose:
                    print(f"Unexpected stop reason: {response.stop_reason}")
                break

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        if final_path:
            if verbose:
                print(f"\n{'='*60}")
                print(f"✓ Maze solved successfully!")
                print(f"  Path length: {len(final_path)} waypoints")
                print(f"  Iterations used: {iteration}/{max_iterations}")
                print(f"{'='*60}\n")
            return final_path
        else:
            print(f"LLM Tool Solver: Failed to find path after {iteration} iterations")
            return None

    except Exception as e:
        print(f"LLM Tool Solver Error: {e}")
        import traceback
        traceback.print_exc()
        return None

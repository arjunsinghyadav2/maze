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
            model="claude-3-5-sonnet-20241022", 
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

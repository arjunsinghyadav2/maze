import cv2
import numpy as np
from collections import deque
import heapq

def is_valid_position(maze, pos):
    """
    Check if a position is valid (within bounds and not a wall).

    Args:
        maze: Binary maze image (0 = wall, 255 = path)
        pos: (x, y) tuple

    Returns:
        True if valid, False otherwise
    """
    x, y = pos
    h, w = maze.shape

    if x < 0 or x >= w or y < 0 or y >= h:
        return False

    # Check if it's a path (white pixel)
    return maze[y, x] > 128


def get_neighbors(pos):
    """
    Get 4-connected neighbors of a position.

    Args:
        pos: (x, y) tuple

    Returns:
        List of neighboring positions
    """
    x, y = pos
    # 4-connected neighbors (up, down, left, right)
    neighbors = [
        (x, y - 1),  # up
        (x, y + 1),  # down
        (x - 1, y),  # left
        (x + 1, y),  # right
    ]
    return neighbors


def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two points."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance(pos1, pos2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def a_star_search(maze, start, goal, verbose=False, goal_radius=100):
    """
    A* pathfinding algorithm to find path from start to goal.

    If goal is blocked, finds path to within goal_radius pixels and
    adds a straight line to the actual goal.

    Args:
        maze: Binary maze image (0 = wall, 255 = path)
        start: (x, y) starting position
        goal: (x, y) goal position
        verbose: If True, print debug information
        goal_radius: Distance within which we consider goal reached (default: 100)

    Returns:
        List of (x, y) positions from start to goal, or None if no path
    """
    if not is_valid_position(maze, start):
        print(f"  ✗ Start position {start} is not valid (on wall or out of bounds)!")
        return None

    # Check if goal is valid - if not, we'll path to within goal_radius
    goal_is_valid = is_valid_position(maze, goal)

    if verbose:
        print(f"  ✓ Start position {start} is valid")
        if goal_is_valid:
            print(f"  ✓ Goal position {goal} is valid")
        else:
            print(f"  ⚠ Goal position {goal} is blocked - will path to within {goal_radius}px")
        print(f"  Manhattan distance: {manhattan_distance(start, goal)}")

    # Priority queue: (f_score, counter, position)
    # counter ensures FIFO ordering for equal f_scores
    counter = 0
    open_set = [(0, counter, start)]
    counter += 1

    # Track where each position came from
    came_from = {}

    # g_score: cost from start to position
    g_score = {start: 0}

    # f_score: g_score + heuristic
    f_score = {start: manhattan_distance(start, goal)}

    # Keep track of positions in open set for faster lookup
    open_set_hash = {start}

    # Track explored positions
    explored = 0
    closest_point = start
    closest_distance = euclidean_distance(start, goal)

    while open_set:
        # Get position with lowest f_score
        current_f, _, current = heapq.heappop(open_set)
        open_set_hash.discard(current)
        explored += 1

        # Track closest point to goal
        dist_to_goal = euclidean_distance(current, goal)
        if dist_to_goal < closest_distance:
            closest_distance = dist_to_goal
            closest_point = current

        # Check if we reached the exact goal OR within goal_radius
        if current == goal or dist_to_goal <= goal_radius:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            # If we didn't reach exact goal, add straight line to goal
            if path[-1] != goal:
                if verbose:
                    print(f"  ✓ Reached within {goal_radius}px of goal (distance: {dist_to_goal:.1f}px)")
                    print(f"  Adding straight line from {path[-1]} to {goal}")
                # Add intermediate points along straight line for smoother visualization
                last_point = path[-1]
                dx = goal[0] - last_point[0]
                dy = goal[1] - last_point[1]
                num_steps = max(abs(dx), abs(dy))
                if num_steps > 0:
                    for i in range(1, num_steps + 1):
                        t = i / num_steps
                        x = int(last_point[0] + dx * t)
                        y = int(last_point[1] + dy * t)
                        path.append((x, y))
                # Ensure goal is the final point
                if path[-1] != goal:
                    path.append(goal)

            if verbose:
                print(f"  ✓ Path found! Explored {explored} positions")
            return path

        # Check all neighbors
        for neighbor in get_neighbors(current):
            if not is_valid_position(maze, neighbor):
                continue

            # Cost to reach neighbor through current
            tentative_g_score = g_score[current] + 1

            # If this is a better path to neighbor
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f = tentative_g_score + manhattan_distance(neighbor, goal)
                f_score[neighbor] = f

                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1
                    open_set_hash.add(neighbor)

    # No path found
    if verbose:
        print(f"  ✗ No path found after exploring {explored} positions")
        print(f"  Closest point reached: {closest_point} (distance: {closest_distance:.1f}px)")
    return None


def simplify_path(path, step_size=5):
    """
    Simplify the path by taking every Nth point to reduce robot movements.

    Args:
        path: List of (x, y) positions
        step_size: Take every Nth point

    Returns:
        Simplified list of positions
    """
    if not path:
        return path

    # Always include start and end
    simplified = [path[0]]

    for i in range(step_size, len(path), step_size):
        simplified.append(path[i])

    # Make sure the last point is included
    if simplified[-1] != path[-1]:
        simplified.append(path[-1])

    return simplified


def visualize_path(image, path, start_pos, goal_pos):
    """
    Visualize the path on the maze image.

    Args:
        image: Original BGR image
        path: List of (x, y) positions
        start_pos: Starting position
        goal_pos: Goal position

    Returns:
        Image with path drawn
    """
    vis_image = image.copy()

    if path is None:
        cv2.putText(vis_image, "NO PATH FOUND!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis_image

    # Draw the path with a thicker line for visibility
    for i in range(len(path) - 1):
        cv2.line(vis_image, path[i], path[i + 1], (255, 0, 255), 3)

    # Draw waypoints
    for i, pos in enumerate(path):
        cv2.circle(vis_image, pos, 3, (0, 255, 255), -1)
        if i % 10 == 0:  # Label every 10th point
            cv2.putText(vis_image, str(i), (pos[0] + 5, pos[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Mark start and goal
    cv2.circle(vis_image, start_pos, 8, (0, 0, 255), 2)
    cv2.putText(vis_image, "START", (start_pos[0] - 30, start_pos[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.circle(vis_image, goal_pos, 8, (0, 255, 0), 2)
    cv2.putText(vis_image, "GOAL", (goal_pos[0] - 25, goal_pos[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Add text showing path info
    path_info = f"Path length: {len(path)} points"
    cv2.putText(vis_image, path_info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis_image

import cv2
import numpy as np

def detect_red_circle(image):
    """
    Detect red circle in the image.

    Args:
        image: BGR image

    Returns:
        (x, y) tuple of circle center, or None if not found
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red has two ranges in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour (assuming it's the circle)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the center of the contour
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy)


def detect_other_circle(image, red_pos, min_distance=50, min_area=100):
    """
    Detect the other circle (non-red) in the image by finding colored regions.
    Prioritizes green circles (broad range) to avoid detecting noise.

    Args:
        image: BGR image
        red_pos: (x, y) position of the red circle to exclude
        min_distance: Minimum distance from red circle
        min_area: Minimum contour area to consider

    Returns:
        (x, y) tuple of circle center, or None if not found
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # First, try to find a green circle specifically (broad range)
    # This helps avoid detecting small noisy circles outside the maze
    lower_green = np.array([30, 40, 40])  # Broad green range
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Remove the red circle area
    if red_pos:
        cv2.circle(green_mask, red_pos, 30, 0, -1)

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Find green contours
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check for valid green circles
    green_candidates = []
    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Check distance from red circle
        if red_pos:
            dist = np.sqrt((cx - red_pos[0])**2 + (cy - red_pos[1])**2)
            if dist < min_distance:
                continue

        green_candidates.append((area, (cx, cy)))

    # If we found green circle(s), return the largest one
    if green_candidates:
        green_candidates.sort(key=lambda x: x[0], reverse=True)
        return green_candidates[0][1]

    # If no green circle found, fall back to any saturated color
    # But use stricter area threshold to avoid noise
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([180, 255, 255])
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # Remove the red circle area
    if red_pos:
        cv2.circle(color_mask, red_pos, 30, 0, -1)

    # Apply morphological operations
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour that's far enough from the red circle
    # Use stricter area threshold for fallback (3x larger)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area * 3:  # Stricter threshold for non-green circles
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Check distance from red circle
        if red_pos:
            dist = np.sqrt((cx - red_pos[0])**2 + (cy - red_pos[1])**2)
            if dist < min_distance:
                continue

        valid_contours.append((area, (cx, cy)))

    if not valid_contours:
        return None

    # Return the largest valid contour
    valid_contours.sort(key=lambda x: x[0], reverse=True)
    return valid_contours[0][1]


def detect_circles(image, color='red'):
    """
    Legacy function for backward compatibility.
    Use detect_red_circle() and detect_other_circle() instead.
    """
    if color == 'red':
        return detect_red_circle(image)
    else:
        # This shouldn't be used anymore, but kept for compatibility
        return None


def preprocess_maze(image, wall_clearance=5, debug=False):
    """
    Preprocess the maze image to extract the maze structure.
    Uses edge detection with morphological operations.

    Args:
        image: BGR image of the maze
        wall_clearance: Number of pixels to add as safety margin around walls (default: 5)
        debug: If True, show intermediate images

    Returns:
        Binary maze image (0 = wall, 255 = path)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel edge detection (X and Y gradients)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Normalize to 0-255 range
    sobel_normalized = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255)

    # Threshold to get binary edges
    _, edges = cv2.threshold(sobel_normalized, 50, 255, cv2.THRESH_BINARY)

    # Connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Dilate to thicken walls - THIS IS THE PERFECT STAGE (walls=white)
    walls_white = cv2.dilate(edges_closed, kernel, iterations=2)

    # Invert: walls=0 (black), paths=255 (white) for maze solver
    binary = cv2.bitwise_not(walls_white)

    # Remove colored circles from the maze (make them paths)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
    color_mask = cv2.dilate(color_mask, np.ones((15, 15), np.uint8), iterations=1)
    binary[color_mask > 0] = 255  # Set circles to path

    # Add wall clearance if needed
    if wall_clearance > 0:
        clearance_kernel = np.ones((wall_clearance * 2 + 1, wall_clearance * 2 + 1), np.uint8)
        inverted = cv2.bitwise_not(binary)
        dilated_walls = cv2.dilate(inverted, clearance_kernel, iterations=1)
        binary = cv2.bitwise_not(dilated_walls)

    if debug:
        cv2.imshow("1. Original", image)
        cv2.imshow("2. Grayscale", gray)
        cv2.imshow("3. Blurred", blurred)
        cv2.imshow("4. Sobel Edges", sobel_normalized)
        cv2.imshow("5. Sobel Thresholded", edges)
        cv2.imshow("6. Edges Closed", edges_closed)
        cv2.imshow("7. Edges Closed & Dilated (WALLS=WHITE) ***PERFECT***", walls_white)
        cv2.imshow("8. Inverted (WALLS=0, PATHS=255)", binary)

        print("\n" + "="*60)
        print("MAZE PREPROCESSING PIPELINE - SOBEL METHOD")
        print("="*60)
        print(f"\nStage 7 (Edges Closed & Dilated) is the perfect wall detection")
        print(f"  White pixels (walls): {np.sum(walls_white == 255)}")
        print(f"  Black pixels (paths): {np.sum(walls_white == 0)}")
        print(f"\nFinal binary maze (for solver):")
        print(f"  Walls (0): {np.sum(binary == 0)}")
        print(f"  Paths (255): {np.sum(binary == 255)}")
        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return binary


def visualize_detection(image, red_pos, other_pos):
    """
    Visualize detected circles on the image.

    Args:
        image: Original BGR image
        red_pos: (x, y) position of red circle
        other_pos: (x, y) position of other circle

    Returns:
        Image with circles marked
    """
    vis_image = image.copy()

    if red_pos:
        cv2.circle(vis_image, red_pos, 10, (0, 0, 255), 2)
        cv2.putText(vis_image, "RED", (red_pos[0] - 20, red_pos[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if other_pos:
        cv2.circle(vis_image, other_pos, 10, (255, 0, 255), 2)
        cv2.putText(vis_image, "OTHER", (other_pos[0] - 30, other_pos[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return vis_image

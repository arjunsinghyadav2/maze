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

    Args:
        image: BGR image of the maze
        wall_clearance: Number of pixels to add as safety margin around walls (default: 5)
        debug: If True, show intermediate images

    Returns:
        Binary maze image (0 = wall, 255 = path)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu's thresholding (works best for this case)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Check if we need to invert (walls should be black/0, paths should be white/255)
    h, w = binary.shape
    center_region = binary[h//4:3*h//4, w//4:3*w//4]
    center_mean = np.mean(center_region)

    # If center is dark, we probably need to invert
    if center_mean < 128:
        binary = cv2.bitwise_not(binary)

    # Close gaps in walls with morphological closing
    kernel_closing = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_closing, iterations=2)

    # Remove small noise (background speckles outside maze)
    kernel_small = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)

    # Remove circles from the binary image to avoid interference
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mask for any saturated colors (circles)
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([180, 255, 255])
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # Dilate the color mask to ensure we remove the entire circle
    kernel = np.ones((15, 15), np.uint8)
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)

    # Set colored areas to white (path) in the binary image
    binary[color_mask > 0] = 255

    # Store the binary before clearance for debugging
    binary_before_clearance = binary.copy()

    # Add wall clearance by dilating walls (eroding paths)
    # This creates a safety margin so the robot doesn't bump into walls
    if wall_clearance > 0:
        # Create a kernel for dilation
        clearance_kernel = np.ones((wall_clearance * 2 + 1, wall_clearance * 2 + 1), np.uint8)

        # Invert to make walls white, paths black
        inverted = cv2.bitwise_not(binary)

        # Dilate the walls (make them thicker)
        dilated_walls = cv2.dilate(inverted, clearance_kernel, iterations=1)

        # Invert back to get the final binary maze
        binary = cv2.bitwise_not(dilated_walls)

    if debug:
        cv2.imshow("1. Original", image)
        cv2.imshow("2. Grayscale", gray)
        cv2.imshow("3. Otsu Threshold", binary_before_clearance)
        cv2.imshow("4. Color Mask (circles)", color_mask)
        if wall_clearance > 0:
            cv2.imshow("5. Final with Clearance", binary)
        print("\nMaze preprocessing debug:")
        print(f"  Using Otsu's thresholding (auto-calculated threshold)")
        print(f"  Center region mean brightness: {center_mean:.1f}")
        print(f"  Morphological operations applied:")
        print(f"    - MORPH_CLOSE (5x5, 2 iterations) to connect wall gaps")
        print(f"    - MORPH_OPEN (3x3, 2 iterations) to remove background noise")
        print(f"  Final: Black pixels (walls): {np.sum(binary == 0)}")
        print(f"  Final: White pixels (paths): {np.sum(binary == 255)}")
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

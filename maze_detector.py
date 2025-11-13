import cv2
import numpy as np

def detect_circles(image, color='red'):
    """
    Detect red or green circles in the image.

    Args:
        image: BGR image
        color: 'red' or 'green'

    Returns:
        (x, y) tuple of circle center, or None if not found
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if color == 'red':
        # Red has two ranges in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == 'green':
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
    else:
        raise ValueError("Color must be 'red' or 'green'")

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


def preprocess_maze(image, debug=False):
    """
    Preprocess the maze image to extract the maze structure.

    Args:
        image: BGR image of the maze
        debug: If True, show intermediate images

    Returns:
        Binary maze image (0 = wall, 255 = path)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Invert if needed (walls should be black/0, paths should be white/255)
    # Check which color is more dominant to determine if we need to invert
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)

    if black_pixels > white_pixels:
        binary = cv2.bitwise_not(binary)

    # Remove circles from the binary image to avoid interference
    # Create a mask to remove red and green areas
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red mask
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Green mask
    lower_green = np.array([40, 30, 30])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Combine masks
    color_mask = cv2.bitwise_or(red_mask, green_mask)

    # Dilate the color mask to ensure we remove the entire circle
    kernel = np.ones((15, 15), np.uint8)
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)

    # Set colored areas to white (path) in the binary image
    binary[color_mask > 0] = 255

    if debug:
        cv2.imshow("Original", image)
        cv2.imshow("Gray", gray)
        cv2.imshow("Binary", binary)
        cv2.imshow("Color Mask", color_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return binary


def visualize_detection(image, red_pos, green_pos):
    """
    Visualize detected circles on the image.

    Args:
        image: Original BGR image
        red_pos: (x, y) position of red circle
        green_pos: (x, y) position of green circle

    Returns:
        Image with circles marked
    """
    vis_image = image.copy()

    if red_pos:
        cv2.circle(vis_image, red_pos, 10, (0, 0, 255), 2)
        cv2.putText(vis_image, "RED", (red_pos[0] - 20, red_pos[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if green_pos:
        cv2.circle(vis_image, green_pos, 10, (0, 255, 0), 2)
        cv2.putText(vis_image, "GREEN", (green_pos[0] - 30, green_pos[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return vis_image

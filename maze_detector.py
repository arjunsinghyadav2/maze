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
    Detect the other circle (non-red) by finding all circles and picking
    the one with the highest green content.

    Args:
        image: BGR image
        red_pos: (x, y) position of the red circle to exclude
        min_distance: Minimum distance from red circle
        min_area: Minimum contour area to consider

    Returns:
        (x, y) tuple of circle center, or None if not found
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Find ALL saturated colored regions (potential circles)
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([180, 255, 255])
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # Remove the red circle area
    if red_pos:
        cv2.circle(color_mask, red_pos, 30, 0, -1)

    # Apply morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    # Find all contours
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Define green range for measuring green content
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Evaluate each contour for green content
    candidates = []
    for contour in contours:
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

        # Calculate green content in this contour
        # Create a mask for this specific contour
        contour_mask = np.zeros(color_mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        # Count green pixels within this contour
        green_in_contour = cv2.bitwise_and(green_mask, contour_mask)
        green_pixel_count = np.sum(green_in_contour > 0)

        # Calculate green percentage in this contour
        green_percentage = green_pixel_count / area if area > 0 else 0

        candidates.append({
            'position': (cx, cy),
            'area': area,
            'green_pixels': green_pixel_count,
            'green_percentage': green_percentage
        })

    if not candidates:
        return None

    # Sort by green pixel count (highest green content first)
    candidates.sort(key=lambda x: x['green_pixels'], reverse=True)

    # Debug output
    print(f"  Found {len(candidates)} circle candidates:")
    for i, c in enumerate(candidates[:3]):  # Show top 3
        print(f"    {i+1}. Position: {c['position']}, Area: {c['area']:.0f}, "
              f"Green pixels: {c['green_pixels']}, Green %: {c['green_percentage']*100:.1f}%")

    # Return the circle with the most green content
    return candidates[0]['position']


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


def preprocess_maze(image, wall_clearance=5, debug=False, start_pos=None, goal_pos=None, circle_radius=45):
    """
    Preprocess the maze image to extract the maze structure.
    Uses enhanced edge detection with CLAHE, bilateral filtering, and Canny edge detection.

    Args:
        image: BGR image of the maze
        wall_clearance: Number of pixels to add as safety margin around walls (default: 5)
        debug: If True, show intermediate images
        start_pos: (x, y) position of start circle (optional, for localized circle removal)
        goal_pos: (x, y) position of goal circle (optional, for localized circle removal)
        circle_radius: Radius around start/goal to remove circles (default: 45 pixels)

    Returns:
        Binary maze image (0 = wall, 255 = path)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 2: Bilateral filter - reduces noise while preserving edges better than Gaussian
    bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # Step 3: Canny edge detection - superior to Sobel for crisp edges
    # Use automatic threshold calculation based on image statistics
    median = np.median(bilateral)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))

    # Apply Canny with calculated thresholds
    edges = cv2.Canny(bilateral, lower, upper, apertureSize=3, L2gradient=True)

    # Step 4: Morphological operations to connect broken edges and fill gaps
    # Use optimized kernels for better line connectivity
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Step 5: Dilate to thicken walls for crisp, continuous lines
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    walls_white = cv2.dilate(edges_closed, kernel_dilate, iterations=2)

    # Invert: walls=0 (black), paths=255 (white) for maze solver
    binary = cv2.bitwise_not(walls_white)

    # Store binary before removing circles
    binary_before_circle_removal = binary.copy()

    # Remove colored circles ONLY around start and goal positions (localized removal)
    if start_pos is not None or goal_pos is not None:
        # Create mask for colored regions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
        color_mask_dilated = cv2.dilate(color_mask, np.ones((15, 15), np.uint8), iterations=1)

        # Create a localized mask - only remove circles near start/goal
        localized_mask = np.zeros_like(color_mask_dilated)

        if start_pos is not None:
            cv2.circle(localized_mask, start_pos, circle_radius, 255, -1)

        if goal_pos is not None:
            cv2.circle(localized_mask, goal_pos, circle_radius, 255, -1)

        # Only remove colors within the localized regions
        final_color_mask = cv2.bitwise_and(color_mask_dilated, localized_mask)
        binary[final_color_mask > 0] = 255  # Set circles to path
    else:
        # Fallback: remove all colored circles (old behavior)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
        color_mask = cv2.dilate(color_mask, np.ones((15, 15), np.uint8), iterations=1)
        binary[color_mask > 0] = 255

    # Store binary after circle removal but before clearance
    binary_after_circle_removal = binary.copy()

    # Add wall clearance if needed
    if wall_clearance > 0:
        clearance_kernel = np.ones((wall_clearance * 2 + 1, wall_clearance * 2 + 1), np.uint8)
        inverted = cv2.bitwise_not(binary)
        dilated_walls = cv2.dilate(inverted, clearance_kernel, iterations=1)
        binary = cv2.bitwise_not(dilated_walls)

    if debug:
        cv2.imshow("1. Original", image)
        cv2.imshow("2. Grayscale", gray)
        cv2.imshow("3. CLAHE Enhanced", enhanced)
        cv2.imshow("4. Bilateral Filtered", bilateral)
        cv2.imshow("5. Canny Edges", edges)
        cv2.imshow("6. Edges Closed", edges_closed)
        cv2.imshow("7. Edges Closed & Dilated (WALLS=WHITE) ***CRISP***", walls_white)
        cv2.imshow("8. Inverted (WALLS=0/BLACK, PATHS=255/WHITE)", binary_before_circle_removal)

        # Show localized circle removal mask if positions provided
        if start_pos is not None or goal_pos is not None:
            # Visualize the localized removal regions
            vis_localized = binary_before_circle_removal.copy()
            vis_localized = cv2.cvtColor(vis_localized, cv2.COLOR_GRAY2BGR)
            if start_pos is not None:
                cv2.circle(vis_localized, start_pos, circle_radius, (0, 255, 0), 2)
            if goal_pos is not None:
                cv2.circle(vis_localized, goal_pos, circle_radius, (255, 0, 0), 2)
            cv2.imshow("8b. Localized Circle Removal Zones", vis_localized)

        cv2.imshow("9. After Circle Removal", binary_after_circle_removal)
        if wall_clearance > 0:
            cv2.imshow("10. After Wall Clearance", binary)

        print("\n" + "="*60)
        print("MAZE PREPROCESSING PIPELINE - ENHANCED CANNY METHOD")
        print("="*60)
        print(f"\nCanny thresholds: lower={lower}, upper={upper} (auto-calculated)")
        print(f"CLAHE applied with clipLimit=2.0, tileGridSize=(8,8)")
        print(f"Bilateral filter: d=9, sigmaColor=75, sigmaSpace=75")
        if start_pos is not None or goal_pos is not None:
            print(f"\nCircle removal: LOCALIZED (radius={circle_radius}px)")
            if start_pos:
                print(f"  Start position: {start_pos}")
            if goal_pos:
                print(f"  Goal position: {goal_pos}")
        else:
            print(f"\nCircle removal: GLOBAL (entire image)")
        print(f"\nStage 7 (Edges Closed & Dilated) - PERFECT:")
        print(f"  White pixels (walls): {np.sum(walls_white == 255)}")
        print(f"  Black pixels (paths): {np.sum(walls_white == 0)}")
        print(f"\nStage 8 (Inverted - before circle removal):")
        print(f"  Black pixels (walls): {np.sum(binary_before_circle_removal == 0)}")
        print(f"  White pixels (paths): {np.sum(binary_before_circle_removal == 255)}")
        print(f"\nStage 9 (After circle removal):")
        print(f"  Black pixels (walls): {np.sum(binary_after_circle_removal == 0)}")
        print(f"  White pixels (paths): {np.sum(binary_after_circle_removal == 255)}")
        print(f"  Wall pixels PRESERVED: {np.sum(binary_before_circle_removal == 0) - np.sum(binary_after_circle_removal == 0)} removed")
        if wall_clearance > 0:
            print(f"\nStage 10 (After wall clearance):")
            print(f"  Black pixels (walls): {np.sum(binary == 0)}")
            print(f"  White pixels (paths): {np.sum(binary == 255)}")
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

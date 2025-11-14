#!/usr/bin/env python3
"""
Interactive Camera-to-Robot Calibration Script

This script guides you through the calibration process step-by-step:
1. Connects to Dobot robot
2. Moves to calibration positions
3. Captures images at each position
4. Lets you click on the marker in each image
5. Saves calibration data to JSON
6. Computes affine and homography matrices
7. Tells you exactly what to update in which files
"""

import cv2
import numpy as np
import pydobot
import serial.tools.list_ports
import json
import time
import os
from datetime import datetime

# Calibration positions (robot X, Y, Z coordinates)
# These are example positions - feel free to modify them
CALIBRATION_POSITIONS = [
    {"name": "Home", "x": 250, "y": 0, "z": -45, "r": 0},
    {"name": "Front-Left", "x": 330, "y": 50, "z": -45, "r": 0},
    {"name": "Front-Right", "x": 330, "y": -90, "z": -45, "r": 0},
    {"name": "Mid-Left", "x": 280, "y": 25, "z": -45, "r": 0},
    {"name": "Mid-Center", "x": 280, "y": -30, "z": -45, "r": 0},
    {"name": "Mid-Right", "x": 280, "y": -85, "z": -45, "r": 0},
    {"name": "Back-Center", "x": 220, "y": -5, "z": -45, "r": 0},
]

# Camera settings (matching open_camera.py)
CAM_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Global variables for mouse callback
clicked_point = None
current_image = None

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to capture clicked point"""
    global clicked_point, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        # Draw a circle at clicked point
        if current_image is not None:
            img_copy = current_image.copy()
            cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(img_copy, (x, y), 10, (0, 255, 0), 2)
            cv2.putText(img_copy, f"({x}, {y})", (x + 15, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Click on the marker position", img_copy)

def find_dobot_port():
    """Find the Dobot USB port"""
    print("\n" + "="*60)
    print("FINDING DOBOT USB PORT")
    print("="*60)

    ports = list(serial.tools.list_ports.comports())

    print("\nAvailable serial ports:")
    for i, port in enumerate(ports):
        print(f"  {i+1}. {port.device} - {port.description}")

    # Try to find /dev/tty.usb* automatically
    dobot_port = None
    for port in ports:
        if 'usb' in port.device.lower() or 'tty.usb' in port.device.lower():
            dobot_port = port.device
            print(f"\n✓ Auto-detected Dobot port: {dobot_port}")
            break

    if not dobot_port:
        print("\nCould not auto-detect Dobot port.")
        choice = input("Enter port number or full path (e.g., /dev/tty.usbserial-1440): ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(ports):
            dobot_port = ports[int(choice) - 1].device
        else:
            dobot_port = choice

    return dobot_port

def connect_dobot(port):
    """Connect to Dobot robot"""
    print(f"\nConnecting to Dobot at {port}...")
    try:
        device = pydobot.Dobot(port=port, verbose=False)
        print("✓ Connected to Dobot successfully!")
        return device
    except Exception as e:
        print(f"✗ Failed to connect to Dobot: {e}")
        return None

def setup_camera():
    """Setup camera with correct resolution"""
    print("\n" + "="*60)
    print("SETTING UP CAMERA")
    print("="*60)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"✗ Failed to open camera {CAM_INDEX}")
        return None

    # Set camera resolution (matching open_camera.py)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Test capture
    ret, frame = cap.read()
    if not ret:
        print("✗ Failed to read from camera")
        cap.release()
        return None

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✓ Camera initialized: {actual_width}x{actual_height}")

    if actual_width != CAM_WIDTH or actual_height != CAM_HEIGHT:
        print(f"⚠ Warning: Requested {CAM_WIDTH}x{CAM_HEIGHT} but got {actual_width}x{actual_height}")

    return cap

def move_robot_to_position(device, pos_dict):
    """Move robot to specified position"""
    print(f"\nMoving robot to: {pos_dict['name']}")
    print(f"  Position: X={pos_dict['x']}, Y={pos_dict['y']}, Z={pos_dict['z']}, R={pos_dict['r']}")

    device.speed(50, 50)
    device.move_to(x=pos_dict['x'], y=pos_dict['y'], z=pos_dict['z'], r=pos_dict['r'])
    time.sleep(2)  # Wait for robot to settle

    print("  ✓ Robot moved to position")

def capture_and_click(cap, pos_index, total_positions):
    """Capture image and let user click on marker"""
    global clicked_point, current_image

    print(f"\n--- Calibration Point {pos_index + 1}/{total_positions} ---")

    # Capture image
    print("Capturing image...")
    time.sleep(0.5)  # Brief pause
    ret, frame = cap.read()

    if not ret:
        # Retry once
        ret, frame = cap.read()
        if not ret:
            print("✗ Failed to capture image")
            return None

    current_image = frame.copy()
    clicked_point = None

    # Show image and wait for click
    print("\n" + "="*60)
    print("CLICK ON THE MARKER POSITION IN THE IMAGE")
    print("="*60)
    print("Instructions:")
    print("  1. A window will open showing the camera image")
    print("  2. Click on the center of the calibration marker")
    print("  3. The clicked position will be highlighted")
    print("  4. Press ENTER to confirm, or ESC to recapture")
    print("="*60)

    cv2.namedWindow("Click on the marker position")
    cv2.setMouseCallback("Click on the marker position", mouse_callback)
    cv2.imshow("Click on the marker position", frame)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and clicked_point is not None:  # ENTER
            print(f"✓ Pixel coordinates: {clicked_point}")
            cv2.destroyAllWindows()
            return clicked_point

        elif key == 27:  # ESC
            print("⚠ Recapturing image...")
            clicked_point = None
            ret, frame = cap.read()
            if ret:
                current_image = frame.copy()
                cv2.imshow("Click on the marker position", frame)

        elif key == ord('q'):
            print("⚠ Calibration cancelled by user")
            cv2.destroyAllWindows()
            return None

def fit_affine(img_pts, rob_xy):
    """Fit affine transformation"""
    M, inliers = cv2.estimateAffine2D(
        img_pts.reshape(-1, 1, 2),
        rob_xy.reshape(-1, 1, 2),
        ransacReprojThreshold=1.0,
        refineIters=1000
    )
    if M is None:
        raise RuntimeError("Affine estimation failed")
    return M

def fit_homography(img_pts, rob_xy):
    """Fit homography transformation"""
    H, inliers = cv2.findHomography(img_pts, rob_xy, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if H is None:
        raise RuntimeError("Homography estimation failed")
    return H

def rms_error_affine(M, img_pts, rob_xy):
    """Calculate RMS error for affine"""
    ones = np.ones((img_pts.shape[0], 1))
    uv1 = np.hstack([img_pts, ones])
    pred = (uv1 @ M.T)
    err = rob_xy - pred
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

def rms_error_homography(H, img_pts, rob_xy):
    """Calculate RMS error for homography"""
    uv1 = np.hstack([img_pts, np.ones((img_pts.shape[0], 1))])
    proj = (uv1 @ H.T)
    proj_xy = proj[:, :2] / proj[:, 2:3]
    err = rob_xy - proj_xy
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

def main():
    print("\n" + "="*60)
    print("INTERACTIVE CAMERA-TO-ROBOT CALIBRATION")
    print("="*60)
    print("\nThis script will guide you through the calibration process.")
    print(f"We will collect {len(CALIBRATION_POSITIONS)} calibration points.")
    print("\nMake sure you have:")
    print("  1. Dobot robot connected via USB")
    print("  2. Camera positioned and focused")
    print("  3. A visible calibration marker (e.g., red circle sticker)")
    print("     attached to the robot end-effector")

    input("\nPress ENTER to begin...")

    # Find and connect to Dobot
    port = find_dobot_port()
    device = connect_dobot(port)

    if device is None:
        print("\n✗ Cannot proceed without Dobot connection")
        return

    # Setup camera
    cap = setup_camera()
    if cap is None:
        print("\n✗ Cannot proceed without camera")
        device.close()
        return

    # Move to home position first
    print("\n" + "="*60)
    print("MOVING TO HOME POSITION")
    print("="*60)
    device.speed(50, 50)
    device.home()
    print("✓ Robot is at home position")
    time.sleep(2)

    # Collect calibration data
    calibration_data = []

    for i, pos_dict in enumerate(CALIBRATION_POSITIONS):
        print("\n" + "#"*60)

        # Move robot
        move_robot_to_position(device, pos_dict)

        # Get actual robot position
        time.sleep(0.5)
        x, y, z, r, j1, j2, j3, j4 = device.pose()
        print(f"  Actual robot pose: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

        # Capture image and get pixel coordinates
        pixel_coords = capture_and_click(cap, i, len(CALIBRATION_POSITIONS))

        if pixel_coords is None:
            print("\n⚠ Calibration cancelled")
            cap.release()
            device.close()
            return

        # Save data
        calibration_data.append({
            "point_name": pos_dict['name'],
            "robot_x": float(x),
            "robot_y": float(y),
            "robot_z": float(z),
            "robot_r": float(r),
            "pixel_u": pixel_coords[0],
            "pixel_v": pixel_coords[1]
        })

        print(f"✓ Calibration point {i+1}/{len(CALIBRATION_POSITIONS)} saved")
        print(f"  Robot: ({x:.2f}, {y:.2f})")
        print(f"  Pixel: {pixel_coords}")

    # Clean up hardware
    cap.release()
    device.close()
    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print("COMPUTING CALIBRATION MATRICES")
    print("="*60)

    # Extract arrays
    img_pts = np.array([[d['pixel_u'], d['pixel_v']] for d in calibration_data], dtype=np.float64)
    rob_xy = np.array([[d['robot_x'], d['robot_y']] for d in calibration_data], dtype=np.float64)

    print(f"\nPixel coordinates:\n{img_pts}")
    print(f"\nRobot coordinates:\n{rob_xy}")

    # Fit transformations
    M = fit_affine(img_pts, rob_xy)
    H = fit_homography(img_pts, rob_xy)

    # Calculate errors
    aff_rms = rms_error_affine(M, img_pts, rob_xy)
    hom_rms = rms_error_homography(H, img_pts, rob_xy)

    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)

    print("\nAffine Matrix M (2x3):")
    print(M)

    print("\nHomography Matrix H (3x3):")
    print(H)

    print(f"\nRMS Error (Affine):     {aff_rms:.6f} mm")
    print(f"RMS Error (Homography): {hom_rms:.6f} mm")

    if aff_rms < 3.0:
        print("\n✓ Excellent calibration! RMS error < 3mm")
    elif aff_rms < 5.0:
        print("\n✓ Good calibration. RMS error < 5mm")
    else:
        print("\n⚠ Warning: RMS error is high. Consider recalibrating.")

    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"calibration_{timestamp}.json"

    output_data = {
        "timestamp": timestamp,
        "camera_resolution": {
            "width": CAM_WIDTH,
            "height": CAM_HEIGHT
        },
        "calibration_points": calibration_data,
        "affine_matrix": M.tolist(),
        "homography_matrix": H.tolist(),
        "rms_error_affine": aff_rms,
        "rms_error_homography": hom_rms,
        "pixel_points": img_pts.tolist(),
        "robot_points": rob_xy.tolist()
    }

    with open(json_filename, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "="*60)
    print("CALIBRATION DATA SAVED")
    print("="*60)
    print(f"\n✓ Calibration saved to: {json_filename}")

    # Print update instructions
    print("\n" + "="*60)
    print("NEXT STEPS - UPDATE YOUR CODE")
    print("="*60)

    print("\n1. Open the file: main_maze_solver.py")
    print("\n2. Find the Affine matrix definition (around line 25):")
    print("   M = np.array([")
    print("       [...],")
    print("       [...],")
    print("   ], dtype=np.float64)")

    print("\n3. Replace it with:")
    print("   M = np.array([")
    for row in M:
        print(f"       {row.tolist()},")
    print("   ], dtype=np.float64)")

    print("\n4. Find the Homography matrix definition (around line 32):")
    print("   H = np.array([")
    print("       [...],")
    print("       [...],")
    print("       [...],")
    print("   ], dtype=np.float64)")

    print("\n5. Replace it with:")
    print("   H = np.array([")
    for row in H:
        print(f"       {row.tolist()},")
    print("   ], dtype=np.float64)")

    print("\n" + "="*60)
    print("CALIBRATION COMPLETE!")
    print("="*60)
    print(f"\nCalibration data: {json_filename}")
    print("Follow the steps above to update main_maze_solver.py")
    print("\nYou can also update Affine_transform.py if needed:")
    print(f"  - Update img_pts (lines 9-17) with: {img_pts.tolist()}")
    print(f"  - Update rob_xy (lines 19-27) with: {rob_xy.tolist()}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Calibration interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error during calibration: {e}")
        import traceback
        traceback.print_exc()

import numpy as np
import cv2  
from pydobot.dobot import MODE_PTP
import time
import pydobot
from camera_utilities import apply_affine, fit_affine, apply_homography, fit_homography
from robot_utilities import move_to_home, move_to_specific_position, get_current_pose

M = np.array([
    [ 1.12811771e-03, -1.82296980e-01,  3.73759663e+02],
    [-1.77124686e-01, -2.58143768e-03,  1.46987989e+02]], dtype=np.float64)

H = np.array([
    [-2.44594058e-02, -4.75669460e-01,  3.67247188e+02],
    [-4.34041615e-01,  5.08065338e-03,  1.20901686e+02],
    [-5.98330506e-05 ,-7.62411614e-05 , 1.00000000e+00]]
    , dtype=np.float64)

def move_robot_point(device,M,u,v):
    Xa, Ya = apply_affine(M, u, v) # Using Affine
    # Xa, Ya = apply_homography(H, u, v) # Using Homography
    print(f"Affine:  pixel({u:.3f}, {v:.3f}) -> robot({Xa:.6f}, {Ya:.6f})")
    move_to_specific_position(device, x=Xa, y=Ya, z=-25)
    time.sleep(1)

def main():
    device = pydobot.Dobot(port="/dev/tty.usbmodem479631A314332")
    device.speed(50, 50)
    move_to_home(device)
    time.sleep(2)

    # Example pixel coordinates from clicks
    pixel_coords = [
        (462, 78),
        (398, 77),
        (395, 145),
        (271, 142),
        (270, 206),
        (201, 210),
        (204, 328),]
    
    for (u, v) in pixel_coords:
        move_robot_point(device ,M, u, v) 

    device.close() 
    
if __name__ == "__main__":
    main()

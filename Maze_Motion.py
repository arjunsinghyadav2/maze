import numpy as np
import cv2  
from pydobot.dobot import MODE_PTP
import time
import pydobot
from camera_utilities import apply_affine, fit_affine, apply_homography, fit_homography
from robot_utilities import move_to_home, move_to_specific_position, get_current_pose

M = np.array([
     [-7.03946756e-03 ,-4.69080162e-01,  3.69730111e+02],
     [-4.47296787e-01  ,5.97834246e-03,  1.24360926e+02]
], dtype=np.float64)

H = np.array([
    [-2.44594058e-02, -4.75669460e-01,  3.67247188e+02],
    [-4.34041615e-01,  5.08065338e-03,  1.20901686e+02],
    [-5.98330506e-05 ,-7.62411614e-05 , 1.00000000e+00]]
    , dtype=np.float64)

def move_robot_point(device,M,u,v):
    Xa, Ya = apply_affine(M, u, v) # Using Affine
    # Xa, Ya = apply_homography(H, u, v) # Using Homography
    print(f"Affine:  pixel({u:.3f}, {v:.3f}) -> robot({Xa:.6f}, {Ya:.6f})")
    move_to_specific_position(device, x=Xa, y=Ya, z=-45)
    time.sleep(1)

def main():
    device = pydobot.Dobot(port="/dev/ttyACM0")
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

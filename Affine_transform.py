# filename: pixel_to_robot_mapper.py
import numpy as np
import cv2

# ================================
# 1) Put your calibration pairs here
#    Image (u,v)  ->  Robot (X,Y)
# ================================
img_pts = np.array([
    [393, 280],
    [642, 287],
    [884, 280],
    [399, 532],
    [640, 532],
    [879, 528],
    [401, 767],
    [644, 772],
    [885, 769],
    [774, 881],
    [537, 140],
], dtype=np.float64)

rob_xy = np.array([
    [321.5728759765625,  74.36516571044922],
    [322.33294677734375, 32.72563552856445],
    [320.9270935058594,  -11.661246299743652],
    [277.0953369140625, 75.10476684570312],
    [277.6030578613281,  31.741823196411133],
    [278.1905212402344, -10.108365058898926],
    [236.07965087890625, 73.64484405517578],
    [235.6623992919922,  29.928909301757812],
    [234.74041748046875, -11.55410385131836],
    [213.1710968017578, 5.652870178222656],
    [347.4062805175781, 48.6517448425293]
], dtype=np.float64)

def fit_affine(img_pts, rob_xy):
    """Fit affine [X Y]^T = M * [u v 1]^T using OpenCV."""
    M, inliers = cv2.estimateAffine2D(
        img_pts.reshape(-1,1,2),
        rob_xy.reshape(-1,1,2),
        ransacReprojThreshold=1.0,
        refineIters=1000
    )
    if M is None:
        raise RuntimeError("Affine estimation failed. Points may be degenerate.")
    return M

def fit_homography(img_pts, rob_xy):
    """Fit projective H so that [X Y 1]^T ~ H * [u v 1]^T."""
    H, inliers = cv2.findHomography(img_pts, rob_xy, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if H is None:
        raise RuntimeError("Homography estimation failed. Points may be degenerate.")
    return H

def apply_affine(M, u, v):
    """Apply affine transform (2x3) to a single pixel (u,v) -> (X,Y)."""
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    XY = M @ uv1
    return float(XY[0]), float(XY[1])

def apply_homography(H, u, v):
    """Apply homography (3x3) to a single pixel (u,v) -> (X,Y)."""
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    Xp, Yp, W = H @ uv1
    if abs(W) < 1e-12:
        raise ZeroDivisionError("Homography scale ~ 0 for this point.")
    return float(Xp / W), float(Yp / W)

def rms_error_affine(M, img_pts, rob_xy):
    ones = np.ones((img_pts.shape[0], 1))
    uv1 = np.hstack([img_pts, ones])          # (N,3)
    pred = (uv1 @ M.T)                        # (N,2)
    err = rob_xy - pred
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

def rms_error_homography(H, img_pts, rob_xy):
    uv1 = np.hstack([img_pts, np.ones((img_pts.shape[0],1))])  # (N,3)
    proj = (uv1 @ H.T)                                         # (N,3)
    proj_xy = proj[:, :2] / proj[:, 2:3]
    err = rob_xy - proj_xy
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

def main():
    # --- Fit both models ---
    M = fit_affine(img_pts, rob_xy)
    H = fit_homography(img_pts, rob_xy)

    print("Affine matrix M (2x3):\n", M)
    print("\nHomography H (3x3):\n", H)

    # --- Report RMS fit error ---
    aff_rms = rms_error_affine(M, img_pts, rob_xy)
    hom_rms = rms_error_homography(H, img_pts, rob_xy)
    print(f"\nRMS error (affine):    {aff_rms:.6f} (robot units)")
    print(f"RMS error (homography): {hom_rms:.6f} (robot units)")
if __name__ == "__main__":
    main()

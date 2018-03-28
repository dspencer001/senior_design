import math
import cv2
import numpy as np
from scipy.signal import medfilt
import track_ic.ellipse_lib as el

def fit_ellipse(center_coords, gray, iris_max, iris_min):
    ellipse_radius = np.linspace(iris_min, iris_max, 40)
    radii = np.unique(np.round(ellipse_radius))
    angles = np.linspace(0, 2 * math.pi, 40)

    # Make more efficient, find out how to not go over the entire image
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

    candidate_radii = []

    for theta in angles:
        temp_best_mag = -1;
        temp_best_r = -1;

        for r in radii:
            pt = np.round([
                center_coords[0] + r * math.sin(theta),
                center_coords[1] + r * math.cos(theta)]).astype(int)

            g_mag = math.sqrt(
                    gx[pt[0]][pt[1]] * gx[pt[0]][pt[1]] +
                    gy[pt[0]][pt[1]] * gy[pt[0]][pt[1]])
            g_hat = (gx[pt[0]][pt[1]]/ g_mag, gy[pt[0]][pt[1]]/ g_mag)

            r_vec = (r * math.cos(theta), r * math.sin(theta))

            if temp_best_mag < g_mag:
                temp_best_mag = g_mag
                temp_best_r = r

        candidate_radii.append(temp_best_r)

    candidate_points = []

    print(candidate_radii)
    print("======================")
    filtered_radii = medfilt(candidate_radii, 5)
    print(filtered_radii)

    for angle, mag in zip(angles, filtered_radii):
        pt = np.round([
            center_coords[0] + mag * math.sin(angle),
            center_coords[1] + mag * math.cos(angle)]).astype(int)

        candidate_points.append(pt)

    new_mat = []

    cp_transpose = np.transpose(candidate_points)

    new_mat.append(cp_transpose[1])
    new_mat.append(cp_transpose[0])

    lsqe = el.LSqEllipse()
    lsqe.fit(new_mat)

    return lsqe.parameters()





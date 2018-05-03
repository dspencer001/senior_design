import random
import math
import cv2
import numpy as np
from scipy.signal import medfilt
import track_ic.ellipse_lib as el

from track_ic.ellipse_model import EllipseModel
from track_ic.ransac import ransac

def get_candidate_points(center_coords, iris_max, iris_min, gx, gy):
    shape = np.shape(gx)
    ellipse_radius = np.linspace(iris_min, iris_max, 40)
    radii = np.unique(np.round(ellipse_radius))
    angles = np.linspace(0, 2 * math.pi, 40)

    # Make more efficient, find out how to not go over the entire image

    candidate_radii = []

    for theta in angles:
        temp_best_mag = -1;
        temp_best_r = -1
        r_dot_g_thresh = 1
        g_mag_threshold = 10
        for r in radii:
            #print(r)
            pt = np.round([
                center_coords[0] + r * math.sin(theta),
                center_coords[1] + r * math.cos(theta)]).astype(int)

            if pt[0] < 0 or pt[0] >= shape[0] or pt[1] < 0 or pt[1] >= shape[1]:
                break

            g_mag = math.sqrt(
                    gx[pt[0]][pt[1]] * gx[pt[0]][pt[1]] +
                    gy[pt[0]][pt[1]] * gy[pt[0]][pt[1]])
            g_hat = (gx[pt[0]][pt[1]]/ g_mag, gy[pt[0]][pt[1]]/ g_mag)

            if g_mag < g_mag_threshold:
                continue

            r_vec = (r * math.cos(theta), r * math.sin(theta))
            r_dot_g = np.dot(r_vec, g_hat)
            if r_dot_g < r_dot_g_thresh:
                continue
            if temp_best_mag < g_mag:
                temp_best_mag = g_mag
                temp_best_r = r

        if temp_best_r > 0:
            candidate_radii.append(temp_best_r)

    candidate_points = []

    filtered_radii = medfilt(candidate_radii, 5)

    for angle, mag in zip(angles, filtered_radii):
        pt = np.round([
            center_coords[0] + mag * math.sin(angle),
            center_coords[1] + mag * math.cos(angle)]).astype(int)

        candidate_points.append(pt)

    return candidate_points


def fit_ellipse(center_coords, gray, iris_max, iris_min):
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

    candidate_points = get_candidate_points(center_coords, iris_max, iris_min, gx, gy)
    rand_pt = random.sample(candidate_points, 5)

    candidate_points = np.unique(candidate_points, axis=0)
    #print(candidate_points)

    new_mat = []
   # return candidate_points
   #RANSAC
    ellipse_model = EllipseModel(gx, gy, gray)
    ransac_fit, ransac_data = ransac(
        candidate_points, ellipse_model,
        5, 30, 100, 20,
        return_all=True)

    return ransac_fit.parameters()




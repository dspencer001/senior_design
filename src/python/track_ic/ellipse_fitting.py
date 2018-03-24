import math
import cv2
import numpy as np
def fit_ellipse(center_coords, gray, iris_max, iris_min):
    print(center_coords)
    ellipse_radius = np.linspace(iris_min, iris_max, 40)
    radii = np.unique(np.round(ellipse_radius))
    
    # Make more efficient, find out how to not go over the entire image
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

    candidate_points = 0
    temp_best = [0, 0]
    for theta in np.linspace(0, 2*math.pi, 40):
        for r in radii:
            pt = np.round([center_coords[0] + r * math.sin(theta), center_coords[1] + r * math.cos(theta)]).astype(int)
            g_mag = math.sqrt(gx[pt[0]][pt[1]] * gx[pt[0]][pt[1]] + gy[pt[0]][pt[1]] * gy[pt[0]][pt[1]])  
            g = (gx[pt[0]][pt[1]]/ g_mag, gy[pt[0]][pt[1]]/ g_mag)
            print(g)    
    
import sys
import math
import numpy as np
import scipy.ndimage.filters as filters
from scipy.signal import medfilt

sys.path.append('/usr/local/lib/python3.6/site-packages')

from track_ic.ellipse_fitting import *
from track_ic.corner_detection import *

from pandas import *

import cv2

# Load an color image in grayscale
img = cv2.imread('david_eye_2.jpg')
#capture = cv2.VideoCapture(2)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


iris_radius = 28.0

rad_min = iris_radius - iris_radius * 0.10
rad_max = iris_radius + iris_radius * 0.10
iris_min = rad_min * rad_min
iris_max = rad_max * rad_max

# Should be a bit larger than 2x the iris radius max, and preferably odd.
kernel_size = 60

oca = np.zeros((kernel_size, kernel_size), complex)
wa = np.zeros((kernel_size, kernel_size))

bounds = int((kernel_size - 1) / 2)

for m in range(-1 * bounds, bounds + 1):
    for n in range(-1 * bounds, bounds + 1):
        if iris_max > (m*m + n*n):
            c = 0
            if m > 0 or n > 0:
                c = 1/math.sqrt(m*m + n*n)

            wa[m + bounds][n + bounds] = c

            if (iris_min < (m*m + n*n)):
                theta = 0
                if n == 0:
                    theta = math.pi / 2 if m > 0 else -1 * math.pi / 2
                else:
                    theta = math.atan(m/n)

                oca[m + bounds][n + bounds] = complex(math.cos(theta) * c, math.sin(theta) * c)

print(oca)


beta = 2
crcc = beta * cv2.Scharr(oca.real, cv2.CV_64F, 1, 0) + (1/beta) * cv2.Scharr(oca.imag, cv2.CV_64F, 0, 1)


lam = 0.5
weighted_img = (1 - lam) * cv2.filter2D(255 - gray, cv2.CV_64F, wa)
filtered_img = lam * cv2.filter2D(gray, cv2.CV_64F, crcc)
co = filtered_img + weighted_img

# adjust this value down for darker images
threshold = 1800

data_max = filters.maximum_filter(co, 11)
data_min = filters.minimum_filter(co, 11)
diff = ((data_max - data_min) > threshold)
maxima = (co == data_max)
maxima[diff == 0] = 0
max_coords = np.transpose(np.where(maxima))

abs_max = co.max()
abs_maxima = (co == abs_max)
abs_coords = np.transpose(np.where(abs_maxima))
print(abs_max)

max_psr = -1
max_psr_coords = []

for coords in max_coords:
    row = coords[0]
    col = coords[1]
    print(coords)
    cv2.rectangle(img,(col, row),(col + 2,row + 2),(0,0,255),2)
    roi_max = co[row-5:row+6, col-5:col+6]
    mu = roi_max.mean()
    dev = roi_max.std()
    #print(mu)
    #print(dev)
    psr = (co[row][col] - mu) / dev
    if psr > max_psr:
        max_psr = psr
        max_psr_coords = coords

row = max_psr_coords[0]
col = max_psr_coords[1]

print(row)
print(col)

c_image = corner_detect(gray, img, row, col)

#center, width, height, phi = fit_ellipse((row, col), gray, rad_max, rad_min)
##candidate_points = fit_ellipse((row, col), gray, rad_max, rad_min)
##for pt in candidate_points:
##    cv2.rectangle(img, (pt[1],pt[0]), (pt[1] + 1, pt[0] + 1), (0,255,0), 1)
#
#
##cv2.ellipse(img, (center[1], center[0]), (width / 2, height / 2), phi, (0, 360), 255, 2)
#pixel_center = np.round(center).astype(int)
#width = round(width).astype(int)
#height = round(height).astype(int)
#
#cv2.ellipse(img,(pixel_center[0], pixel_center[1]),(width, height),phi,0,360,(0,255,0),1)
#
#
#cv2.rectangle(img,(col, row),(col + 2,row + 2),(0,255,0),2)
#cv2.rectangle(img,(pixel_center[0], pixel_center[1]),(pixel_center[0] + 2,pixel_center[1] + 2),(0,0,255),2)
#
##cv2.rectangle(img,(abs_coords[0][1], abs_coords[0][0]), (abs_coords[0][1] + 2, abs_coords[0][0] + 2), (255,255,0), 2)
#print(max_psr)

#cv2.imshow('image',img)
cv2.imshow('image', c_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


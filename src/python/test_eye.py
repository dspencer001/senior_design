import sys
import math
import numpy as np
import scipy.ndimage.filters as filters
from pandas import *
import cv2

# Load an color image in grayscale
img = cv2.imread('Eye.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

iris_radius = 33.0

iris_min = iris_radius - iris_radius * 0.2
iris_max = iris_radius + iris_radius * 0.2
iris_min = iris_min * iris_min
iris_max = iris_max * iris_max

kernel_size = 45
oca = np.zeros((kernel_size, kernel_size), complex)
wa = np.zeros((kernel_size, kernel_size))

for m in range(1, kernel_size + 1):
    for n in range(1, kernel_size + 1):
        if iris_max > (m*m + n*n):
            c = 1/math.sqrt(m*m + n*n)
            wa[m-1][n-1] = c
            if (iris_min < (m*m + n*n)):
                theta = math.atan(n/m)
                oca[m-1][n-1] = complex(math.cos(theta) * c, math.sin(theta) * c)


beta = 1.5
crcc = beta * cv2.Scharr(oca.real, cv2.CV_64F, 1, 0) + (1/beta) * cv2.Scharr(oca.imag, cv2.CV_64F, 0, 1)


lam = 0.9
weighted_img = (1 - lam) * cv2.filter2D(255 - gray, cv2.CV_64F, wa)
filtered_img = lam * cv2.filter2D(gray, cv2.CV_64F, crcc)
c = filtered_img + weighted_img

threshold = 300
data_max = filters.maximum_filter(c, 11)
data_min = filters.minimum_filter(c, 11)
diff = ((data_max - data_min) > threshold)
maxima = (c == data_max)
maxima[diff == 0] = 0
max_coords = np.transpose(np.where(maxima))


for coords in max_coords:
    row = coords[0]
    col = coords[1]
    print(coords)
    cv2.rectangle(img,(col, row),(col + 2,row + 2),(0,0,255),2)
    roi_max = c[row-5:row+6, col-5:col+6]
    mu = roi_max.mean()
    dev = roi_max.std()
    print(mu)
    print(dev)
    psr = (c[row][col] - mu) / dev
    print(psr)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


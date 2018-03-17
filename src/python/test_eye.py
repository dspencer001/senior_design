import sys
import math
import numpy as np
import scipy.ndimage.filters as filters
from pandas import *

sys.path.append('/usr/local/lib/python3.6/site-packages')

import cv2
lam = 0.3
weighted_img = (1 - lam) * cv2.filter2D(255 - eye_roi_gray, cv2.CV_64F, wa)
filtered_img = lam * cv2.filter2D(eye_roi_gray, cv2.CV_64F, crcc)
c = filtered_img + weighted_img

threshold = 200
data_max = filters.maximum_filter(c, 11)
data_min = filters.minimum_filter(c, 11)
diff = ((data_max - data_min) > threshold)
maxima = (c == data_max)
maxima[diff == 0] = 0
max_coords = np.transpose(np.where(maxima))

for coords in max_coords:
    row = coords[0]
    col = coords[1]
    cv2.rectangle(eye_roi_color,(col, row),(col + 2,row + 2),(0,0,255),2)
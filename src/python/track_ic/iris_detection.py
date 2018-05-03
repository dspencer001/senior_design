import sys
import math
import numpy as np
import scipy.ndimage.filters as filters
from scipy.signal import medfilt

sys.path.append('/usr/local/lib/python3.6/site-packages')

from track_ic.ellipse_fitting import *

from pandas import *

import cv2

faceCascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

def find_rois(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+int(h*0.6), x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            minNeighbors=5,
            minSize=(10,10),
            maxSize=(int(w/2), int(w/2))
        )
        for (ex,ey,ew,eh) in eyes:
            if(ew > w / 5):
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]

                find_iris(eye_roi_color, eye_roi_gray, w)


def find_iris(img, gray, face_width):
    #print(gray)

    #iris_radius = len(img[1]) * 0.33 / 2
    iris_radius = face_width / 27

    rad_min = iris_radius - iris_radius * 0.1
    rad_max = iris_radius + iris_radius * 0.1
    iris_min = rad_min * rad_min
    iris_max = rad_max * rad_max

    # Should be a bit larger than 2x the iris radius max, and preferably odd.
    kernel_size = int(iris_radius * 2 + 3)

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


    beta = 2
    crcc = beta * cv2.Scharr(oca.real, cv2.CV_64F, 1, 0) + (1/beta) * cv2.Scharr(oca.imag, cv2.CV_64F, 0, 1)


    #lam = 0.3
    lam= 0.3
    weighted_img = (1 - lam) * cv2.filter2D(255 - gray, cv2.CV_64F, wa)
    filtered_img = lam * cv2.filter2D(gray, cv2.CV_64F, crcc)
    co = filtered_img + weighted_img

# adjust this value down for darker images

    data_max = filters.maximum_filter(co, 11)
    data_min = filters.minimum_filter(co, 11)
    data_diff = (data_max - data_min)
    threshold = 1000
    diff = (data_diff > threshold)
    maxima = (co == data_max)
    maxima[diff == 0] = 0
    max_coords = np.transpose(np.where(maxima))

    abs_max = co.max()
    abs_maxima = (co == abs_max)
    abs_coords = np.transpose(np.where(abs_maxima))


    #print(abs_max)

    #print(max_coords)

    max_psr = -1
    max_psr_coords = []

    for coords in max_coords:
        row = coords[0]
        col = coords[1]
        #print(coords)
        #cv2.rectangle(img,(col, row),(col + 2,row + 2),(0,0,255),2)
        roi_max = co[row-5:row+6, col-5:col+6]
        mu = roi_max.mean()
        dev = roi_max.std()
        #print(mu)
        #print(dev)
        psr = (co[row][col] - mu) / dev
        if psr > max_psr:
            max_psr = psr
            max_psr_coords = coords

    if(len(max_psr_coords) ==  0):
        return

    row = max_psr_coords[0]
    col = max_psr_coords[1]

    #print((row, col))
    #cv2.rectangle(img, (col, row), (col + 2,row + 2),(0,255,0),8)

    try:
        center, width, height, phi = fit_ellipse((row, col), gray, rad_max, rad_min)
    except ValueError:
        print("No ellipse")
        return
    except numpy.linalg.linalg.LinAlgError:
        print("No ellipse")
        return

    center = np.round(center).astype(int)
    width = int(width)
    height = int(height)

#    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
#    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
#
#    candidate_points = get_candidate_points((row, col), rad_max, rad_min, gx, gy)
#    for pt in candidate_points:
#        cv2.rectangle(img, (pt[1],pt[0]), (pt[1] + 1, pt[0] + 1), (0,255,0), 2)


    #print(center)
    #print(width)
    #print(height)

    cv2.ellipse(img, (center[0], center[1]), (width, height), int(np.rad2deg(phi)), 0, 360, (0, 255, 0), 2)

    cv2.rectangle(img,(center[0], center[1]),(center[0] + 2,center[1] + 2),(0,255,0),2)

import sys
import math
import numpy as np
import scipy.ndimage.filters as filters
from pandas import *

sys.path.append('/usr/local/lib/python3.6/site-packages')

import cv2

cascPath = sys.argv[1]
print(cv2.getBuildInformation())

faceCascade = cv2.CascadeClassifier(cascPath)
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

video_capture = cv2.VideoCapture(1)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 3);
video_capture.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

ex_1 = None;
ex_2 = None;

ey_1 = None;
ey_2 = None;

while True:
    # Capture frame-by-frame
    #for i in range(4):
    #    video_capture.grab()

    ret, frame = video_capture.read()
    weighted_img = frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    convolvedx = cv2.Scharr(gray, -1, 1, 0)

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
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        iris_radius = 0.042 * w

        iris_min = iris_radius - iris_radius * 0.1
        iris_max = iris_radius + iris_radius * 0.1
        iris_min = iris_min * iris_min
        iris_max = iris_max * iris_max

        kernel_size = 20
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


        beta = 3
        crcc = beta * cv2.Scharr(oca.real, cv2.CV_64F, 1, 0) + (1/beta) * cv2.Scharr(oca.imag, cv2.CV_64F, 0, 1)

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            minNeighbors=5,
            minSize=(10,10)
        )
        for (ex,ey,ew,eh) in eyes:
            eye_roi_gray = gray[ey:ey+eh, ex:ex+ew]
            eye_roi_color = frame[ey:ey+eh, ex:ex+ew]

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


            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

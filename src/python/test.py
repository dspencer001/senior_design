import sys
import math
import numpy as np
import scipy.ndimage.filters as filters
from pandas import *
from track_ic.find_iris import *

sys.path.append('/usr/local/lib/python3.6/site-packages')

import cv2

faceCascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

video_capture = cv2.VideoCapture(2)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 3);
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);

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

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            minNeighbors=5,
            minSize=(10,10)
        )
        for (ex,ey,ew,eh) in eyes:
            if(ew > w / 5):
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]

                find_iris(eye_roi_color, eye_roi_gray)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

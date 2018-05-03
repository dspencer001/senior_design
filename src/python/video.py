import sys
import math
import numpy as np
import scipy.ndimage.filters as filters
from pandas import *
from track_ic.iris_detection import *

sys.path.append('/usr/local/lib/python3.6/site-packages')

import cv2

video_capture = cv2.VideoCapture(2)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 3);
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);

video_capture.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

while True:
    # Capture frame-by-frame
    #for i in range(4):
    #    video_capture.grab()

    ret, frame = video_capture.read()

    find_rois(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2
from track_ic.iris_detection import *

img = cv2.imread('david_face.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

find_rois(img)

cv2.imshow('image',img)
cv2.imwrite('face_out.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

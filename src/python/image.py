import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2
from track_ic.find_iris import *

img = cv2.imread('david_eye_4.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

find_iris(img, gray)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

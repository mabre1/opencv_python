import numpy as np
import cv2 as cv

# read image from file
img = cv.imread('./resources/images/clouds.jpg',0)

# display image
cv.imshow('image',img)

# wait for ESC key to exit
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
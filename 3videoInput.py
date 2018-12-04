import numpy as np
import cv2 as cv 

# this does magic and gives you back a camera object to read from
cap = cv.VideoCapture(0)


while(True):
        # read a frame
        ret, frame = cap.read()
        
        # show the frame
        cv.imshow('frame',frame)

        # exit if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
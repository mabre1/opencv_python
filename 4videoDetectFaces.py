import numpy as np
import cv2 as cv 

# load in a pre-trained classifier
face_cascade = cv.CascadeClassifier('./resources/haarCascades/haarcascade_frontalface_default.xml')

# this does magic and gives you back a camera object to read from
cap = cv.VideoCapture(0)
while(True):
        # read frame
        ret, frame = cap.read()
        
        # this algorithm expects grayscale and is colour independant
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame using our pretained classifier
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # loop over results drawing boxes around faces
        for (x,y,w,h) in faces:
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # show resulting frame
        cv.imshow('faces',frame)

        # quit when 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
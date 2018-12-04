import numpy as np
import cv2 as cv 
print(cv.__version__)

# load in a pre-trained face classifier
face_cascade = cv.CascadeClassifier('./resources/haarCascades/haarcascade_frontalface_default.xml')
# load in a pre-trained eye classifier
eye_cascade = cv.CascadeClassifier('./resources/haarCascades/haarcascade_eye.xml')

# this does magic and gives you back a camera object to read from
cap = cv.VideoCapture(0)
while(True):
        # read frame
        ret, frame = cap.read()

        # this algorithm expects grayscale and is colour independant
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame using our pretained face classifier
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # loop over results
        for (x,y,w,h) in faces:

                # draw boxes around faces
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                # get a 'Region Of Interest' - A subset of the frame representing the area identified to contain a face
                roi_gray = gray[y:y+h, x:x+w]
                # get the same subset from the colour version
                roi_color = frame[y:y+h, x:x+w]

                # detect eyes in said subset - detect eyes only in the detected 'face areas'
                eyes = eye_cascade.detectMultiScale(roi_gray)

                # draw boxes around the detected eyes
                for (ex,ey,ew,eh) in eyes:
                        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        # show resulting frame
        cv.imshow('frame',frame)

        # if 'q' is pressed, exit
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
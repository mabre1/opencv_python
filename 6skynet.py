import numpy as np
import cv2 as cv 

# load in a pre-trained face classifier
face_cascade = cv.CascadeClassifier('./resources/haarCascades/haarcascade_frontalface_default.xml')
# load in a pre-trained eye classifier
eye_cascade = cv.CascadeClassifier('./resources/haarCascades/haarcascade_eye.xml')

# select a font for writing in the image with
font = cv.FONT_HERSHEY_SIMPLEX

# this does magic and gives you back a camera object to read from
cap = cv.VideoCapture(0)
while(True):
        # read frame
        ret, frame = cap.read()

        # this algorithm expects grayscale and is colour independant
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame using our pretained face classifier
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # set count of detected faces to 0
        faceCount = 0
        
        # loop through results
        for (x,y,w,h) in faces:

                # draw boxes around faces
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                
                # Write text to screen, with position and font size reletive to detection size
                cv.putText(frame,'HUMAN DETECTED:',(x,int(y+h+(w/8))), font, (w/300),(66, 149, 244),int(w/75),cv.LINE_AA)
                cv.putText(frame,'ACTIVATE LASERS', (int(x+(w/8)),int(y+h+(w/4))), font, (w/300),(0,0,255),int(w/75),cv.LINE_AA)
                
                # increment the count of detections
                faceCount += 1

                # get a 'Region Of Interest' - A subset of the frame representing the area identified to contain a face
                roi_gray = gray[y:y+h, x:x+w]
                # get the same subset from the colour version
                roi_color = frame[y:y+h, x:x+w]

                # detect eyes in said subset - detect eyes only in the detected 'face areas'
                eyes = eye_cascade.detectMultiScale(roi_gray)

                # draw boxes around the detected eyes
                for (ex,ey,ew,eh) in eyes:
                        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        # display the count of faces
        cv.putText(frame, 'Humans: ' + str(faceCount),(20,30), font, 0.5,(100,255,255),2,cv.LINE_AA)

        # show resulting frame
        cv.imshow('frame',frame)

        # if 'q' is pressed, exit
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
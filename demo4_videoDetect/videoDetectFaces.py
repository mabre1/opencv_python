import numpy as np
import cv2 as cv 

face_cascade = cv.CascadeClassifier('../resources/haarCascades/haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)
while(True):
        ret, frame = cap.read()
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
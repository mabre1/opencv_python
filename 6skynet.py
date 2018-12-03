import numpy as np
import cv2 as cv 
face_cascade = cv.CascadeClassifier('./resources/haarCascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('./resources/haarCascades/haarcascade_eye.xml')
hats = [cv.imread('./resources/images/hat1.jpg',0),cv.imread('./resources/images/hat2.jpg',0),cv.imread('./resources/images/hat3.jpg',0)]
singleHat = cv.imread('./resources/images/hat3.jpg',0)
font = cv.FONT_HERSHEY_SIMPLEX
cap = cv.VideoCapture(0)
while(True):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        faceCount = 0
        
        for (x,y,w,h) in faces:
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                
                cv.putText(frame,'HUMAN DETECTED:',(x,int(y+h+(w/8))), font, (w/300),(66, 149, 244),int(w/75),cv.LINE_AA)
                cv.putText(frame,'ACTIVATE LASERS', (int(x+(w/8)),int(y+h+(w/4))), font, (w/300),(0,0,255),int(w/75),cv.LINE_AA)
                faceCount += 1

                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
        
                for (ex,ey,ew,eh) in eyes:
                        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                                
        cv.putText(frame, 'Humans: ' + str(faceCount),(20,30), font, 0.5,(100,255,255),2,cv.LINE_AA)
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
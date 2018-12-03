import numpy as np
import cv2 as cv
import time
face_cascade = cv.CascadeClassifier('./resources/haarCascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('./resources/haarCascades/haarcascade_eye.xml')
hats = [cv.imread('./resources/images/hat1.jpg',0),cv.imread('./resources/images/hat2.jpg',0),cv.imread('./resources/images/hat3.jpg',0)]
singleHat = cv.imread('./resources/images/hat3.jpg',0)
font = cv.FONT_HERSHEY_SIMPLEX
startTime = time.time()
refreshTime = 2
counter = 0
cap = cv.VideoCapture(0)
while(True):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        faceCount = 0
        
        for (x,y,w,h) in faces:
                # cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                
                faceCount += 1

                ratio = w / singleHat.shape[1]
                hat = cv.resize(singleHat,None,fx=ratio, fy=ratio, interpolation = cv.INTER_CUBIC)

                
                y_offset = y-hat.shape[0]
                x_offset = int((x + (w/2))-(hat.shape[1]/2))
                hat_cutoff_y = hat.shape[0] - (frame.shape[0] - y_offset)
                hat_cutoff_x = hat.shape[1] - (frame.shape[1] - x_offset)

                # draw a hat ontop of all detected faces
                for j in range(0,hat.shape[0]-max(0, hat_cutoff_y)):
                        for i in range(0,hat.shape[1]-max(0, hat_cutoff_x)):
                                # shitty chroma-key: 
                                if hat.item(j,i) < 190:
                                        frame[y_offset+j, x_offset+i] = hat[j,i]

        counter +=1
        cv.putText(frame,'FPS: ' + str(int(counter / (time.time() - startTime))),(20,30), font, 0.5,(100,255,255),2,cv.LINE_AA)
        if (time.time() - startTime) > refreshTime:
                counter = 0
                startTime = time.time()
        
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
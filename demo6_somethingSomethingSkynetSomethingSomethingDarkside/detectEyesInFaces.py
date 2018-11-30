import numpy as np
import cv2 as cv 
print(cv.__version__)
face_cascade = cv.CascadeClassifier('../resources/haarCascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('../resources/haarCascades/haarcascade_eye.xml')
cap = cv.VideoCapture(0)
while(True):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(frame,'DISCO MATT NEEDS BEER',(x-30,y-15), font, 0.5,(115,55,55),2,cv.LINE_8)
                cv.putText(frame,'HUMAN DETECTED:',(x-40,y+h+20), font, 0.5,(66, 149, 244),2,cv.LINE_AA)
                cv.putText(frame,'ACTIVATE LASERS',(x,y+h+40), font, 0.5,(0,0,255),2,cv.LINE_AA)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                # cv.putText(frame,str(eyes.size),(x+60,y-60), font, 0.5,(0,0,255),2,cv.LINE_AA)
                # i = 0
                # points = np.array((2,1))
                for (ex,ey,ew,eh) in eyes:
                        # points = points.append([ex+(ew/2),ey])
                        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                        # cv.putText(roi_color,str(i),(ex+10,ey-10), font, 0.5,(0,0,255),2,cv.LINE_AA)
                        # i += 1

                # first = 1
                # points = np.array((2,2))
                # points = np.append([[x,y]],[[1,2]], axis=0)
                # for (gx,gy) in points:
                #         if first:
                #                 # points = np.arra([[gx,gy]])
                #                 first = 0
                #                 # points[0][0] = gx
                #                 # points[0][1] = gy
                #                 # memy = gy
                #         else:
                #                 cv.line(roi_color,(gx,gy),(points[0][0],points[0][1]),(255,0,0),5)

        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
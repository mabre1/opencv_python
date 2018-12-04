import numpy as np
import cv2 as cv
import time

# load in a pre-trained classifiers
face_cascade = cv.CascadeClassifier('./resources/haarCascades/haarcascade_frontalface_default.xml')
# # load in a hat image
singleHat = cv.imread('./resources/images/hat3.jpg',0)
# select a font for writing in the image with
font = cv.FONT_HERSHEY_SIMPLEX

# set start time for FPS counter
startTime = time.time()
# the fps counter tracks better with regular reset (every 2 seconds)
refreshTime = 2
# set frame count to 0
counter = 0

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
        
        for (x,y,w,h) in faces:
                # increment the count of detections
                faceCount += 1

                # calculate a ratio to draw the hat to a scale that fits the head - near or far
                ratio = w / singleHat.shape[1]

                # scale hat based on said ratio
                hat = cv.resize(singleHat,None,fx=ratio, fy=ratio, interpolation = cv.INTER_CUBIC)

                # some calculations to align the hat with the top center of the detected faces
                # set an offset for y axis - this makes the bottom of the hat line up with the top of your hear
                y_offset = y-hat.shape[0]
                # set an offset for x axis - this makes the horozontal center of the detected face
                x_offset = int((x + (w/2))-(hat.shape[1]/2))
                # these two lines prevent out of bounds conditions if the hat goes off the side of the screen
                hat_cutoff_y = hat.shape[0] - (frame.shape[0] - y_offset)
                hat_cutoff_x = hat.shape[1] - (frame.shape[1] - x_offset)

                # draw a hat ontop of all detected faces
                for j in range(0,hat.shape[0]-max(0, hat_cutoff_y)):
                        for i in range(0,hat.shape[1]-max(0, hat_cutoff_x)):
                                # shitty chroma-key - if the colour is less 75% white, draw the pixel (there are better ways to do this)
                                if hat.item(j,i) < 190:
                                        frame[y_offset+j, x_offset+i] = hat[j,i]
        # increment frame counter
        counter +=1
        # display FPS counter
        cv.putText(frame,'FPS: ' + str(int(counter / (time.time() - startTime))),(20,30), font, 0.5,(100,255,255),2,cv.LINE_AA)
        
        # refresh counter if refresh time (in seconds) has expired reset FPS counter
        if (time.time() - startTime) > refreshTime:
                counter = 0
                startTime = time.time()
        
        # show resulting image
        cv.imshow('frame',frame)

        # if 'q' is pressed, exit
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
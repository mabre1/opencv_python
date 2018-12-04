import numpy as np
import cv2 as cv

# read in a pretrained classifier from file
face_cascade = cv.CascadeClassifier('./resources/haarCascades/haarcascade_frontalface_default.xml')

# read image from file
img = cv.imread('./resources/images/FaceBase.jpg')

# convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# detect faces in image using classifier
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# loop over results and draw rectangles around the faces
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# display results image
cv.imshow('result',img)

# wait for keypress to exit
cv.waitKey(0)
cv.destroyAllWindows()

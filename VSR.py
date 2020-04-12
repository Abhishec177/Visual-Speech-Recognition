#Import necessary libraries
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

#CAPTURE VIDEO, INITIALIZE GLOBAL VARIABLES
cap = cv2.VideoCapture(0)
pts_inner = []
pts_outer = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    ret, frame = cap.read()
    
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
        
        landmarks = predictor(gray,face)
        
        #MAP OUTER BOUNDARY OF THE LIPS 
        for i in range(48,60): 
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            t = (x,y)
            pts_outer.append(t)
            cv2.circle(frame,(x,y),1,(255,255,255),-1)
    
    
        pts_outer = np.array(pts_outer,np.int32)
        pts_outer.reshape((-1,1,2))
        cv2.polylines(frame,[pts_outer],True,(255,0,255),2)    
        pts_outer = []
    
        #MAP INNER BOUNDARIES OF THE LIPS
        for i in range(60,68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            t = (x,y)
            pts_inner.append(t)
            cv2.circle(frame,(x,y),1,(255,255,255),-1)
    
        pts_inner = np.array(pts_inner,np.int32)
        pts_inner.reshape((-1,1,2))
        cv2.polylines(frame,[pts_inner],True,(255,0,255),2)    
        pts_inner = []
    
    
    cv2.imshow('frame',frame)
    
    #QUIT ON PRESSING Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
#RELEASE CAPTURE AND CLOSE VIDEO STREAM WINDOW       
cap.release()
cv2.destroyAllWindows()
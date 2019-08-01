#!/usr/bin/python3

import cv2 as cv
import time

def GenarateData(img, face_cascade, eye_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgWithSection = img 
    face = face_cascade.detectMultiScale(gray)
    train_gray = []
    for x,y,w,h in face: #face has been detected
        #print('face', face)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eye = eye_cascade.detectMultiScale(roi_gray) # detect eye
        for ex,ey,ew,eh in eye:
            #print('eye', eye)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)

        imgWithSection = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if len(face) == 1:
            train_gray = cv2.resize(roi_gray, (92, 112))

    return train_gray,imgWithSection

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier("/opt/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_alt2.xml")
    eye_cascade = cv2.CascadeClassifier("/opt/opencv-3.4.1/data/haarcascades/haarcascade_eye.xml")
    camera = cv2.VideoCapture(0)
    loop = 10
    i = 0

    while True:
        ret, frame = camera.read()
        train_gray,imgWithSection = GenarateData(frame, face_cascade, eye_cascade)
        cv2.imshow('face', imgWithSection)

        loop = loop -1
        if len(train_gray) > 0 and loop <= 0:
            print('save', i)
            cv2.imwrite(str(i) + '.pgm', train_gray)
            i = i + 1
            loop = 10

        key = cv2.waitKey(30) & 0xff
        if key == 27: # ESC
            break

    camera.release()
    cv2.destroyAllWindows()

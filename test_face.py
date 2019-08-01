#!/usr/bin/python3

import cv2
import os
import sys

modelFile = './face.xml'

def GetXData(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray)
    if len(face) <= 0:
        return [],[] 
    train_gray = []
    for x,y,w,h in face:
        #print('face', face)
        roi_gray = gray[y:y+h, x:x+w]
        if len(face) == 1:
            train_gray = cv2.resize(roi_gray, (92, 112))
    return train_gray,face

def Predict(model):
    face_cascade = cv2.CascadeClassifier("/opt/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_alt2.xml")
    camera = cv2.VideoCapture(0)

    while True:
        succ, frame = camera.read()
        faceImg,face_rect = GetXData(frame, face_cascade)
        if len(faceImg) > 0:
            for x,y,w,h in face_rect:
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            label,acc = model.predict(faceImg)
            print(label, acc)

            if label == 41:
                cv2.putText(frame, 'admin', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 0), 2)
            else:
                cv2.putText(frame, str(label), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 0), 2)

        cv2.imshow('img', frame)
        key = cv2.waitKey(30) & 0xff
        if key == 27: # ESC
            break

def PredictOnePic(model, imgFile):
    print('test', imgFile)
    x = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
    label, accuracy = model.predict(x)
    print(label, accuracy)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        modelName = sys.argv[1]
    else:
        print('usage:{} LBP/Fisher/Eigen'.format(sys.argv[0]))
        sys.exit(1)
    print('modeltype:', modelName)
    if not os.path.exists(modelFile):
        print('modelFile not exists!')
    else:
        if modelName == 'LBP':
            model = cv2.face.LBPHFaceRecognizer_create()
        elif modelName == 'Fisher':
            model = cv2.face.FisherFaceRecognizer_create()
        elif modelName == 'Eigen':
            model = cv2.face.EigenFaceRecognizer_create()
        model.read(modelFile)
        Predict(model)
        #PredictOnePic(model, './orl_faces/s9/5.pgm')

#!/usr/bin/python3

import os
import cv2
import numpy as np
import sys

dataDir = './orl_faces'

def GetTrainData():
    imgs = []
    labels = [] # requires ints
    level1Files = os.listdir(dataDir)
    #print('dirs:', level1Files)

    for level1File in level1Files:
        absDir = dataDir + '/' + level1File
        if os.path.isdir(absDir):
            myid = int(level1File[1:])
            imgFiles = os.listdir(absDir)

            for imgFile in imgFiles:
                img = cv2.imread(absDir + '/' + imgFile, cv2.IMREAD_GRAYSCALE)
                print('img:', absDir + '/' + imgFile, 'label:', myid)
                imgs.append(img)
                labels.append(myid)

    return imgs, labels

def Train(modelName):
    print('train...', modelName)
    # choice model type
    if modelName == 'LBP':
        model = cv2.face.LBPHFaceRecognizer_create()
    elif modelName == 'Fisher':
        model = cv2.face.FisherFaceRecognizer_create()
    elif modelName == 'Eigen':
        model = cv2.face.EigenFaceRecognizer_create()
    else:
        print('miss modelname')
        sys.exit(1)
    imgs, labels = GetTrainData()
    model.train(np.array(imgs), np.array(labels))
    model.save('face.xml')

if __name__ == '__main__':
    if len(sys.argv) == 2:
        modelName = sys.argv[1]
    else:
        print('usage:{} LBP/Fisher/Eigen'.format(sys.argv[0]))
        sys.exit(1)
    Train(modelName)

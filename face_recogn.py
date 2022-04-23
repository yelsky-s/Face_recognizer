import os #module to loop through directories
import cv2 as cv
import numpy as np

#haarcascade images face clasifier
haarcascade_face = cv.CascadeClassifier('haarcascade_face.xml')

#address to the folder of arranged photos of friends
phot_directory=r'C:\Users\yelsk\Desktop\phot'
#creating people's names list from photos folder (same as it was in our training set)
friends_names = []
for i in os.listdir(phot_directory):
    friends_names.append(i)

#loading our previously created faces and labels lists (from faces_train file)
faces=np.load('faces.npy', allow_pickle=True)
labels=np.load('labels.npy')

#loading a face recognizer (from faces_train file)
face_recognizer= cv.face.LBPHFaceRecognizer_create()

#reading our created trained model (from faces_train file) with a face_recognizer
face_recognizer.read('facerecogn_trained.yml')

#image to recognize faces on. validation
img=cv.imread(r'C:\Users\yelsk\Desktop\programming\Projects DS\openCV_study\IMG-20160826.jpg')

gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow('friends', gray_img)

#detect faces in the image
faces_rectangle=haarcascade_face.detectMultiScale(gray_img,1.1,6)

for (x,y,width,height) in faces_rectangle:
    faces_relev_region=gray_img[y:y+height, x:x+width]

    label, confidence = face_recognizer.predict(faces_relev_region)
    print(f'label ={friends_names[label]}, confidence level of {confidence}')

    #placing rectangle and text over recognised face
    cv.putText(img, str(friends_names[label]), (x,y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y),(x+width,y+height), (0,255,0), thickness=2)

cv.imshow('detected faces', img)

cv.waitKey(0)
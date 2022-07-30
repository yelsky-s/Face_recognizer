import os #module to loop through directories
import cv2 as cv
import numpy as np

#address to the folder of arranged photos of friends
phot_directory=r'C:\Users\yelsk\Desktop\phot'

#creating people's names list from photos folder. the photos are arranged into folders for each friend
friends_names = []
for i in os.listdir(phot_directory):
    friends_names.append(i)

#haarcascade images face clasifier
haarcascade_face = cv.CascadeClassifier('haarcascade_face.xml')

#list of features = images of faces
faces=[]
#list of labels= names of people -- indexes corresponding to indexes of faces for
labels=[]

#function to create a database of faces found in each folder of photos with coresponding name of a friend
def database_creat():

    for person in friends_names:
        path=os.path.join(phot_directory, person)
        label = friends_names.index(person)

        for image in os.listdir(path):
            image_path = os.path.join(path, image)

            image_array=cv.imread(image_path)
            gray_conv=cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)

            face_rectangle = haarcascade_face.detectMultiScale(gray_conv, scaleFactor=1.1,minNeighbors=10)

            #looping over every faces found in the gray_conv image to save the face rectangle/s that are found into 'faces'
            for (x,y,width, height) in face_rectangle:
                face_relev_region = gray_conv[y:y+height, x:x+width]

                #append found faces (face_relev_region) to faces and labels lists
                faces.append(face_relev_region)
                labels.append(label)

database_creat()

print("finished training")
#print(f'length of our faces list is {len(faces)}')
#print(f'length of our labels list is {len(labels)}')

#converting faces and labels lists to numpy arrays
faces=np.array(faces, dtype='object')
labels=np.array(labels)

#face recognizer
face_recognizer= cv.face.LBPHFaceRecognizer_create()

#train recognizer on the faces list and labels list
face_recognizer.train(faces, labels)

#saving our trained model so it can be used later --> in face-recogn,py in our case, for ex.
face_recognizer.save('facerecogn_trained.yml')

#saving our lists of found faces and labels
np.save('faces.npy', faces)
np.save('labels.npy', labels)






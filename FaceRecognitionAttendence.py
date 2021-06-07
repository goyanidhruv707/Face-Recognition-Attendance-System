import cv2
import numpy as np
import face_recognition
import os

path = 'TrainingImages'
imageList = []
nameList = []
myList = os.listdir(path)

for name in myList:
    currentImage = cv2.imread(f'{path}/{name}')
    imageList.append(currentImage)
    nameList.append(os.path.splitext(name)[0])
print(nameList)


def getEncodings(imagesList):
    encodedList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)[0]
        encodedList.append(enc)
    return encodedList

encodeList = getEncodings(imageList)
print("Encoding Completed")

webcam = cv2.VideoCapture(1)
webcam.set(3, 480)
webcam.set(4, 640)
while True:
    success, frame = webcam.read()
    imgSmall = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    imgRGB = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    cv2.imshow("Webcam", frame)
    cv2.waitKey(1)
    faceCurrFrame = face_recognition.face_locations(imgRGB)
    encodeCurrFrame = face_recognition.face_encodings(imgRGB, faceCurrFrame)

    for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
        matches = face_recognition.compare_faces(encodeList, encodeFace)
        faceDistance = face_recognition.face_distance(encodeList, encodeFace)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = nameList[matchIndex].upper()
            print(name)

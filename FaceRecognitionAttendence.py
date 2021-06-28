import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

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


def markAttendence(name):
    with open("Attendence.csv", 'r+') as file:
        dataList = file.readlines()
        names = []
        for line in dataList:
            entry = line.split(",")
            names.append(entry[0])
        if name not in names:
            time = datetime.now()
            dateStr = time.strftime('%H:%M:%S')
            file.writelines(f'\n{name},{dateStr}')


encodeList = getEncodings(imageList)
print("Encoding Completed")

webcam = cv2.VideoCapture(1)
webcam.set(3, 480)
webcam.set(4, 640)

while True:
    success, frame = webcam.read()
    imgSmall = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgRGB = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    faceCurrFrame = face_recognition.face_locations(imgRGB)
    encodeCurrFrame = face_recognition.face_encodings(imgRGB, faceCurrFrame)

    for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
        matches = face_recognition.compare_faces(encodeList, encodeFace)
        faceDistance = face_recognition.face_distance(encodeList, encodeFace)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex] and faceDistance[matchIndex] < 0.5100:
            name = nameList[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            markAttendence(name)

    cv2.imshow("Webcam", frame)
    cv2.waitKey(1)

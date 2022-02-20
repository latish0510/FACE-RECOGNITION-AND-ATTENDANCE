import cv2
import numpy
import numpy as np
import face_recognition
import os
from datetime import datetime

#Step 1
path = 'ImagesAttendance'
images=[]
classNames=[]
mylist=os.listdir(path)
print(mylist)
for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findencodings(images):
    encodelist=[]
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist
#Marking Attendance
def markattendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

encodelistknown=findencodings(images)
print('Encoding Complete')

#Step 3

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodelistknown,encodeFace)
        faceDist=face_recognition.face_distance(encodelistknown,encodeFace)
        #print(faceDist)
        matchIndex=np.argmin(faceDist)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1,y2),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)








    cv2.imshow('webcam',img)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
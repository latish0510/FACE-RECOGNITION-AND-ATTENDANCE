import cv2
import numpy as np
import face_recognition
#Step 1
imgFace=face_recognition.load_image_file('#File Name')
imgFace=cv2.cvtColor(imgFace,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file('#FileName')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
#Step 2
faceLoc=face_recognition.face_locations(imgFace)[0]
encodeFace=face_recognition.face_encodings(imgFace)[0]
cv2.rectangle(imgFace,(faceLoc[3],(faceLoc[0])),(faceLoc[1],(faceLoc[2])),(255,0,255),2)


faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],(faceLocTest[0])),(faceLocTest[1],(faceLocTest[2])),(255,0,255),2)


#Step 3
results=face_recognition.compare_faces([encodeFace],encodeTest)
facedist=face_recognition.face_distance([encodeFace],encodeTest)
print(results)
print(facedist)

cv2.putText(imgTest,f'{results}{round(facedist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)




cv2.imshow("Face Musk",imgFace)
cv2.imshow("Face Test",imgTest)
cv2.waitKey(0)
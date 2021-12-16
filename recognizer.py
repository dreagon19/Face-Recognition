import cv2
import os
from PIL import Image

def recognize():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("training.yml")

    names = []  #names of the folder present in dataset


    for users in os.listdir("dataset"):
        names.append(users)

    cap = cv2.VideoCapture(0)

    while True:
        _,img = cap.read()
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(imgGray,
                                                1.1,
                                                4)

        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            id,_ = recognizer.predict(imgGray[y:y+h,x:x+w])
            if id:
                cv2.putText(img,names[id-1],(x,y-4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
            else:
                cv2.putText(img,"unknown",(x,y-4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)    


        cv2.imshow('cam',img)
        if cv2.waitKey(1) == ord('q'):
            break

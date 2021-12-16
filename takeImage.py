import cv2
from pathlib import Path
import dataset as ds 
import recognizer as rec

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
print("Enter the id and name of the person:")
userId = input()
userName = input()

count = 1

def saveImage(img,userName,userId,imgId):
    Path("dataset/{}".format(userName)).mkdir(parents=True,exist_ok=True)
    cv2.imwrite("dataset/{}/{}_{}.jpg".format(userName,userId,imgId),img)

while True:
    _,img = cap.read()
    originalImg = img
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(imgGray,
                                            1.1,
                                            4)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        coords=[x,y,w,h]

    cv2.imshow('cam',img)
    key = cv2.waitKey(1)

    if(key == ord('s')):
        if count <=5:
            roiImg = originalImg[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
            saveImage(roiImg,userName,userId,count)
            count+=1
        else:
            ds.generateDataset()
            rec.recognize()
    elif (key == ord('q')):
        break
          

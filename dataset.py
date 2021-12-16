import os
import cv2
import numpy as np 

from PIL import Image


def generateDataset():
    names = []  #names of the folder present in dataset
    paths = [] #path to each image

    for users in os.listdir("dataset"):
        names.append(users)


    for name in names:
        for image in os.listdir("dataset\{}".format(name)):
            path_str = os.path.join("dataset\{}".format(name),image)
            paths.append(path_str)


    #print(paths)

    faces =[]
    ids =  []

    for img_path in paths:
        image = Image.open(img_path).convert("L")

        imgNp = np.array(image,"uint8")


        id = int(img_path.split("\\")[2].split("_")[0])

        faces.append(imgNp)
        

        
        ids.append(id)

    ids = np.array(ids)

    trainer = cv2.face.LBPHFaceRecognizer_create()

    trainer.train(faces,ids)

    trainer.write("training.yml")

    print("The model dataset Generated...")
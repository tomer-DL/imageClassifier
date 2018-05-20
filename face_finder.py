import os
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def extract_faces(dir, ext):
    if ext[0] != ".":
        ext = "." + ext
    a=1
    for file in os.listdir(dir):
        if file.endswith(ext):
            img = cv2.imread(dir + "\\" + file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            ar = [];
            for(x,y,w,h) in faces:
                print(str(a))
                cv2.imwrite("face" + str(a) + ".jpg", img[y:y+h,x:x+w])
                a += 1
                

def main():
    extract_faces("C:\\Users\\Orly-PC\\PycharmProjects\\imageClassifier\\small-images", "jpg")


if __name__ == "__main__":
    main()

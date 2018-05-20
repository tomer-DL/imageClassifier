import os
import cv2
import numpy as np

def file_generator(root, ext=""):
    for x in os.listdir(root):
        if os.path.isfile(root + "\\" + x) and x.endswith(ext):
            yield root + "\\" + x
        elif os.path.isdir(root + "\\" + x):
            yield from file_generator(root + "\\" + x, ext)

def main():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")        
    a=1
    for filename in file_generator("D:\\dstRoot\\IMAGE", ".jpg"):
        try:
            img = cv2.imread(filename)
            small = cv2.resize(img, None, (0,0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
            faces = face_cascade.detectMultiScale(small, 1.3, 5)
            for(x,y,w,h) in faces:
                if w>=70:
                    res = cv2.resize(small[y:y+h,x:x+w], (70,70), interpolation=cv2.INTER_AREA)
                    cv2.imwrite("faces\\" + str(a) + ".jpg", res)
                    a += 1
        except Exception as e:
            print(str(e))

if __name__ == "__main__":
    main()


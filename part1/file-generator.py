import os
import cv2
import numpy as np
from read_ini import read_section



def file_generator(root, ext=""):
    for x in os.listdir(root):
        if os.path.isfile(root + "\\" + x) and x.endswith(ext):
            yield root + "\\" + x
        elif os.path.isdir(root + "\\" + x):
            yield from file_generator(root + "\\" + x, ext)

def main():
    properties = read_section("part1.ini", "part1")
    face_cascade = cv2.CascadeClassifier(properties["face.classifier"])        
    a=1
    for filename in file_generator(properties["input.images.root"], properties["images.extension"]):
        try:
            img = cv2.imread(filename)
            fx = float(properties["factor.x"])
            fy = float(properties["factor.y"])
            min_face_size = int(properties["face.min.size"])
            small = cv2.resize(img, None, (0,0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
            faces = face_cascade.detectMultiScale(small, 1.3, 5)
            for(x,y,w,h) in faces:
                if w>=min_face_size:
                    res = cv2.resize(small[y:y+h,x:x+w], (min_face_size,min_face_size), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(properties["output.faces.dir"] + str(a) + properties["faces.extension"], res)
                    a += 1
        except Exception as e:
            print(str(e))

if __name__ == "__main__":
    main()


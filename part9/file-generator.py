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
    properties = read_section("part9.ini", "part9")
    face_cascade = cv2.CascadeClassifier(properties["face.classifier"])        
    fx = float(properties["factor.x"])
    fy = float(properties["factor.y"])
    min_face_size = int(properties["face.min.size"])

    a=1
    images_counter = 0
    max_images=100
    for filename in file_generator(properties["input.images.root"], properties["images.extension"]):
        try:
            images_counter += 1
            img = cv2.imread(filename)
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            for(x,y,w,h) in faces:
                if w>=min_face_size:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
                    a += 1
            new_file_name = properties["output.photos.dir"] + filename[filename.rfind('\\')+1:]
            print(new_file_name)
            cv2.imwrite(new_file_name, img)
            if images_counter>=max_images:
                break
        except Exception as e:
            print(str(e))

if __name__ == "__main__":
    main()


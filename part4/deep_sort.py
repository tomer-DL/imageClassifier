import numpy as np
from keras.models import load_model
import cv2
import os
from utils import file_generator
from utils import read_section

properties = read_section("part4.ini", "part4")
model_dir = properties["model.save.dir"]
model_file = properties["model.save.name"]
faces_out_dir = properties["real.faces.out.dir"]
not_faces_out_dir = properties["not.real.faces.out.dir"]
src_dir = properties["faces.dir"]

model = load_model(model_dir+model_file)



def main():
    out_dir = [not_faces_out_dir, faces_out_dir]
    
    for file_name in file_generator(src_dir):
        try:
            img = cv2.imread(file_name)
            img = img.reshape(1,img.shape[0], img.shape[1], img.shape[2])
            
            a = model.predict(img)
            os.rename(file_name, out_dir[int(a[0][0])] + file_name[len(src_dir):])
        except Exception as e:
            print(str(e))
            


if __name__ == "__main__":
    main()




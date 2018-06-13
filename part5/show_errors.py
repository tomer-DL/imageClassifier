import numpy as np
from keras.models import load_model
import cv2
from utils import file_generator
import os
from utils import read_section

properties = read_section("part5.ini", "part5")
model_dir = properties["model.save.dir"]
model_file = properties["model.save.name"]
faces_dir = properties["real.faces.dir"]
not_faces_dir = properties["not.real.faces.dir"]
model = load_model(model_dir+model_file)


def count_errors(header, src_dir, expected_val):
    err_count =0
    for file_name in file_generator(src_dir):
        try:
            img = cv2.imread(file_name)
            img = img.reshape(1,img.shape[0], img.shape[1], img.shape[2])
            a = model.predict(img)
            if(a[0][0] != expected_val):
                err_count +=1
                print(file_name)
        except Exception as e:
            print(str(e))
    print(header + str(err_count))
    



def main():
    count_errors("faces error count: ", faces_dir, 1)
    count_errors("non-faces error count: ", not_faces_dir, 0)
    

if __name__ == "__main__":
    main()




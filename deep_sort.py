import numpy as np
from keras.models import load_model
import cv2
from create_dataset import file_generator
import os

def main():
    out_dir = ["new_other", "new_noa", "new_itay", "new_orly", "new_tomer"]
    model = load_model("faces.h5")
    src_dir = "faces"
    for file_name in file_generator(src_dir):
        try:
            img = cv2.imread(file_name)
            img = img.reshape(1,img.shape[0], img.shape[1], img.shape[2])
            
            a = model.predict(img)
            for i in range(a.shape[1]):
                if(a[0][i] == 1):
                    os.rename(file_name, out_dir[i] + file_name[len(src_dir):])
        except Exception as e:
            print(str(e))
            


if __name__ == "__main__":
    main()




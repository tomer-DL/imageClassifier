import numpy as np
from keras.models import load_model
import cv2
import os
from read_dataset import file_generator
from read_ini import read_section

properties = read_section("part8.ini", "part8")
model_dir = properties["model.save.dir"]
model_file = properties["model.save.name"]
src_dir = properties["faces.dir"]
img_classes = eval(properties["img.classes"])
output_base_dir = properties["output.base.dir"]

model = load_model(model_dir+model_file)



def main():
    
    for file_name in file_generator(src_dir):
        try:
            img = cv2.imread(file_name)
            img = img.reshape(1,img.shape[0], img.shape[1], img.shape[2])
            img = img *1./255
            img = img[..., ::-1]
            a = model.predict(img)
            idx = np.ndarray.argmax(a);
            os.rename(file_name, output_base_dir + img_classes[idx] + "/" + file_name[len(src_dir):])
        except Exception as e:
            print(str(e))
            


if __name__ == "__main__":
    main()




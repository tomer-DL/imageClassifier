import numpy as np
from keras.models import load_model
import cv2
from file_util import file_generator
import os
from read_ini import read_section
from keras.preprocessing.image import ImageDataGenerator

properties = read_section("part7.ini", "part7")
model_dir = properties["model.save.dir"]
model_file = properties["model.save.name"]

validation_root = properties["validation.dir.root"]
classes = eval(properties["img.classes"]) 
print("loading: ", model_dir+model_file)
model = load_model(model_dir+model_file)

datagen_validation = ImageDataGenerator(rescale=1./255)

validation_generator = datagen_validation.flow_from_directory(
    directory=validation_root,
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    classes=classes,
    shuffle=False
    
)


def count_errors(header, src_dir, expected_val):
    err_count =0
    for file_name in file_generator(src_dir):
        try:
            img = cv2.imread(file_name)
            img = img.reshape(1,img.shape[0], img.shape[1], img.shape[2])
            img = img *1./255
            img = img[..., ::-1]

            a = model.predict(img)
            
            if(np.ndarray.argmax(a) != expected_val):
                err_count +=1
                print(file_name, classes[np.ndarray.argmax(a)])
        except Exception as e:
            print(str(e))
    print(header + str(err_count))


def main():
    ar = model.predict_generator(validation_generator, steps=6)
#    print(ar.shape)
#    print(np.ndarray.argmax(ar, axis=1))

    for i in range(len(classes)):
        count_errors(classes[i] + ": ", validation_root + "/" + classes[i], i)

    

if __name__ == "__main__":
    main()




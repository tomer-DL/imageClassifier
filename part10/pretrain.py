import numpy as np
import os
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from read_ini import read_section
base_model = VGG16(include_top=False, input_shape=(150,150,3))
base_model.summary()

properties = read_section("part10.ini", "part10")
train_dir = properties["train_dir"]
valid_dir = properties["valid_dir"]
train_images = int(properties["train.images.count"])
valid_images = int(properties["validation.images.count"])
train_batch_size = int(properties["train.batch.size"])
valid_batch_size = int(properties["validation.batch.size"])
dataset_dir = properties["dataset.dir"]
classes = eval(properties["img.classes"]) 


def extract_features(dir, num_of_images, batch_size):
    datagen = ImageDataGenerator(rescale=1./255)
    features = np.zeros((num_of_images, 4, 4, 512))
    labels = np.zeros((num_of_images))
    generator = datagen.flow_from_directory(
        dir,
        target_size=(150, 150),
        batch_size = batch_size,
        classes = classes
    )
    i=0

    for input_batch, label_batch in generator:
        feature_batch = base_model.predict(input_batch)
        features[i*batch_size: (i+1)*batch_size] = feature_batch
        print(np.ndarray.argmax(label_batch, axis=1))
        labels[i*batch_size: (i+1)*batch_size] = np.ndarray.argmax(label_batch,axis=1)
        i += 1
        if i * batch_size >= num_of_images:
            break
    return features, labels

def main():
    train_X, train_y = extract_features(train_dir, train_images, train_batch_size)
    validaion_X, validation_y = extract_features(valid_dir, valid_images, valid_batch_size)
    np.save(dataset_dir +"/train_x", train_X)
    np.save(dataset_dir +"/train_y", train_y)
    np.save(dataset_dir +"/valid_x", validaion_X)
    np.save(dataset_dir +"/valid_y", validation_y)

if __name__ == "__main__":
    main()
    

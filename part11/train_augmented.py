import numpy as np
import os
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from read_ini import read_section
import matplotlib.pyplot as plt

base_model = VGG16(include_top=False, input_shape=(150,150,3))
base_model.trainable = False

properties = read_section("part11.ini", "part11")
train_dir = properties["train_dir"]
valid_dir = properties["valid_dir"]
train_images = int(properties["train.images.count"])
valid_images = int(properties["validation.images.count"])
train_batch_size = int(properties["train.batch.size"])
valid_batch_size = int(properties["validation.batch.size"])
classes = eval(properties["img.classes"])
model_dir = properties["model.save.dir"]
model_file = properties["model.save.name"]

def main():
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation="softmax"))

    datagen_train = ImageDataGenerator(rescale=1./255,
        rotation_range=6,
        horizontal_flip=True,
        width_shift_range=5,
        height_shift_range=5
    )

    datagen_validation = ImageDataGenerator(rescale=1./255)

    train_generator = datagen_train.flow_from_directory(
        directory=train_dir,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=train_batch_size,
        class_mode="categorical",
        shuffle=True,
        classes=classes
    )

    validation_generator = datagen_validation.flow_from_directory(
        directory=valid_dir,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=valid_batch_size,
        class_mode="categorical",
        classes=classes
    )

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()
     
    history = model.fit_generator(train_generator, steps_per_epoch=24, epochs=100, 
                    verbose=2, validation_data=validation_generator, validation_steps=6)

    model.save(model_dir + model_file)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()


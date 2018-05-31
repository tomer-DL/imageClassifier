import streamingDataset as sd
import numpy as np
np.random.seed(2808)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from read_ini import read_section
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

properties = read_section("part6.ini", "part6")

training_root = properties["training.dir.root"]
validation_root = properties["validation.dir.root"]

model_dir = properties["model.save.dir"]
model_file = properties["model.save.name"]
classes = eval(properties["img.classes"]) 

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    directory=training_root,
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    classes=classes
)

validation_generator = datagen.flow_from_directory(
    directory=validation_root,
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    classes=classes
)

model = Sequential()
model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
 
# 9. Fit model on training data
history = model.fit_generator(train_generator, steps_per_epoch=24, epochs=40, 
                verbose=2, validation_data=validation_generator, validation_steps=6)

print(history.history) 
# 10. Evaluate model on test data
model.save(model_dir + model_file)

import matplotlib.pyplot as plt
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

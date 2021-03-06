import streamingDataset as sd
import numpy as np
np.random.seed(2808)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from utils import read_section
from keras.optimizers import Adam
properties = read_section("part5.ini", "part5")

dirs = [properties["real.faces.dir"], properties["not.real.faces.dir"]]
classes = [1, 0]
dataset = sd.StreamingDataset(dirs, classes)

model_dir = properties["model.save.dir"]
model_file = properties["model.save.name"]

'''
(X_train, X_test), (y_train, y_test) = load_data()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[3], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[3], X_test.shape[1], X_test.shape[2])
    input_shape = (X_train.shape[3], X_train.shape[1], X_train.shape[2])
else:
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

'''

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
model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=0.0005)
model.compile(loss='mean_squared_error',
              optimizer=adam,
              metrics=['accuracy'])
model.summary()
 
# 9. Fit model on training data
history = model.fit_generator(dataset.generate_training(32), steps_per_epoch=242, epochs=15, \
                verbose=2, validation_data=dataset.generate_test(32), validation_steps=60)

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

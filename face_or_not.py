import numpy as np
np.random.seed(2808)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from create_dataset import load_data

(X_train, X_test), (y_train, y_test) = load_data()
print(K.image_data_format())
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

Y_train = np_utils.to_categorical(y_train, 5)
Y_test = np_utils.to_categorical(y_test, 5)

model = Sequential()
print(input_shape)
model.add(Conv2D(32, 3, 3, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
 
# 9. Fit model on training data
history = model.fit(X_train, Y_train, 
          batch_size=32, epochs=15, verbose=2, validation_data=(X_test, Y_test))

print(history.history) 
# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
model.save('faces.h5')
print(score)

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

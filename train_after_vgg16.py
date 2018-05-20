import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from use_VGG16 import load_data

(X_train, X_test), (y_train, y_test) = load_data()

Y_train = np_utils.to_categorical(y_train, 5)
Y_test = np_utils.to_categorical(y_test, 5)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3]))


print(X_train.shape)

model = Sequential()
model.add(Dense(255, activation='relu', input_dim=2048))
model.add(Dropout(0.5))
model.add(Dense(5, activation="softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
 
# 9. Fit model on training data
history = model.fit(X_train, Y_train, 
          batch_size=32, epochs=30, verbose=2, validation_data=(X_test, Y_test))

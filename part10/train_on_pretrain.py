import numpy as np
from read_ini import read_section
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout


properties = read_section("part10.ini", "part10")
dataset_dir = properties["dataset.dir"]
classes = eval(properties["img.classes"]) 
 
def main():
    train_X = np.load(dataset_dir +"/train_x.npy")
    train_y = np.load(dataset_dir +"/train_y.npy")
    valid_X = np.load(dataset_dir +"/valid_x.npy")
    valid_y = np.load(dataset_dir +"/valid_y.npy")

    train_y = to_categorical(train_y, num_classes=len(classes))
    valid_y = to_categorical(valid_y, num_classes=len(classes))
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1]* train_X.shape[2] * train_X.shape[3]))
    valid_X = np.reshape(valid_X, (valid_X.shape[0], valid_X.shape[1]* valid_X.shape[2] * valid_X.shape[3]))
    
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=train_X.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()

    history = model.fit(train_X, train_y, batch_size=32, epochs=30, verbose=2, validation_data=(valid_X, valid_y))

if __name__ == "__main__":
    main()

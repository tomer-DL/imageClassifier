import numpy as np
from keras.applications import VGG16
from create_dataset import load_data

def main():
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(70,70,3))
    conv_base.summary()
    (X_train, X_test), (y_train, y_test) = load_data()
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    new_x = conv_base.predict(X)
    np.save("vgg16_x", new_x)
    np.save("vgg16_y", y)
    #print(a.shape)

def load_data():
    X = np.load("vgg16_x.npy")
    y = np.load("vgg16_y.npy")
    size = y.shape[0]
    indices = np.random.permutation(size)
    eighty = (int)(size*80/100)
    training_idx, test_idx = indices[:eighty], indices[eighty:]
    training_X, test_X = X[training_idx,:,:], X[test_idx,:,:]
    training_y, test_y = y[training_idx], y[test_idx]
    return (training_X, test_X), (training_y, test_y)



if __name__ == "__main__":
    main()

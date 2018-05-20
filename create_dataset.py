import numpy as np
import os
import cv2

def file_generator(root, ext=""):
    for x in os.listdir(root):
        if os.path.isfile(root + "\\" + x) and x.endswith(ext):
            yield root + "\\" + x

def read_dir(ar, y, dir, cur_class):
    a = 0
    for filename in file_generator(dir):
        try:
            img = cv2.imread(filename)
            if(ar.shape[0] == 1 and a == 0):
                ar[0,:,:,:] = img
                a = 1
            else:
                tmp = np.zeros((1,70,70,3))
                tmp[0,:,:,:] = img
                ar = np.concatenate((ar, tmp), axis=0)    
            y = np.append(y, cur_class)       
            
        except Exception as e:
            print(str(e))
    return ar, y

def save_data(X, y):
    np.save("children-X", X)
    np.save("children-y", y)

def load_data():
    X = np.load("children-X.npy")
    y = np.load("children-y.npy")
    size = y.shape[0]
    indices = np.random.permutation(size)
    eighty = (int)(size*80/100)
    training_idx, test_idx = indices[:eighty], indices[eighty:]
    training_X, test_X = X[training_idx,:,:], X[test_idx,:,:]
    training_y, test_y = y[training_idx], y[test_idx]
    return (training_X, test_X), (training_y, test_y)
    
    


def main():
    ar = np.zeros((1,70,70,3))
    y = np.array([])
    ar, y = read_dir(ar, y, "noa", 1)
    ar, y = read_dir(ar, y, "itay", 2)
    ar, y = read_dir(ar, y, "other", 0)
    ar, y = read_dir(ar, y, "orly", 3)
    ar, y = read_dir(ar, y, "tomer", 4)
    save_data(ar, y)
    (training_X, test_X), (training_y, test_y) = load_data()
    print(training_X.shape)
    print(test_X.shape) 
    print(training_y.shape)
    print(test_y.shape) 

if __name__ == "__main__":
    main()

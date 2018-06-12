import numpy as np
import os
import cv2
from utils import read_section
from utils import file_generator

properties = read_section("part2.ini", "part2")
faces_dir = properties["input.images.faces"]
other_dir = properties["input.images.other"]
extension = properties["images.extension"]
image_height = int(properties["face.size.height"])
image_width = int(properties["face.size.width"])
output_dir = properties["output.dataset.dir"]
output_X = output_dir + properties["dataset.file.name.x"]
output_y = output_dir + properties["dataset.file.name.y"]

def read_dir(ar, y, dir, cur_class):
    a = 0
    for filename in file_generator(dir):
        try:
            img = cv2.imread(filename)
            if(ar.shape[0] == 1 and a == 0):
                ar[0,:,:,:] = img
                a = 1
            else:
                tmp = np.zeros((1,image_height,image_width,3))
                tmp[0,:,:,:] = img
                ar = np.concatenate((ar, tmp), axis=0)    
            y = np.append(y, cur_class)       
            
        except Exception as e:
            print(str(e))
    return ar, y

def save_data(X, y):
    np.save(output_X, X)
    np.save(output_y, y)

def load_data():
    X = np.load(output_X)
    y = np.load(output_y)
    size = y.shape[0]
    indices = np.random.permutation(size)
    eighty = (int)(size*80/100)
    training_idx, test_idx = indices[:eighty], indices[eighty:]
    training_X, test_X = X[training_idx,:,:], X[test_idx,:,:]
    training_y, test_y = y[training_idx], y[test_idx]
    return (training_X, test_X), (training_y, test_y)
    
    


def main():
    ar = np.zeros((1,150,150,3))
    y = np.array([])
    ar, y = read_dir(ar, y, faces_dir, 1)
    ar, y = read_dir(ar, y, other_dir, 0)
    save_data(ar, y)
    (training_X, test_X), (training_y, test_y) = load_data()
    print(training_X.shape)
    print(test_X.shape) 
    print(training_y.shape)
    print(test_y.shape) 

if __name__ == "__main__":
    main()

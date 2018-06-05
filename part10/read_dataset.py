import numpy as np
import os
import cv2
from read_ini import read_section


properties = read_section("part3.ini", "part3")
input_dir = properties["dataset.dir"]
input_X = input_dir + properties["dataset.file.name.x"]
input_y = input_dir + properties["dataset.file.name.y"]
 

def load_data():
    X = np.load(input_X)
    y = np.load(input_y)
    size = y.shape[0]
    indices = np.random.permutation(size)
    eighty = (int)(size*80/100)
    training_idx, test_idx = indices[:eighty], indices[eighty:]
    training_X, test_X = X[training_idx,:,:], X[test_idx,:,:]
    training_y, test_y = y[training_idx], y[test_idx]
    return (training_X, test_X), (training_y, test_y)
    
    
def file_generator(root, ext=""):
    for x in os.listdir(root):
        if os.path.isfile(root + "/" + x) and x.endswith(ext):
            yield root + "/" + x



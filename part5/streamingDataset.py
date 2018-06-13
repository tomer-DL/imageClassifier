from utils import file_generator
import numpy as np
import cv2

class StreamingDataset:
    def __init__(self, dirs, classes):
        self.X = []
        self.y = []
        for i in range(len(dirs)):
            self.read_dir(dirs[i], classes[i])
        self.create_datasets()

    def read_dir(self, dir, cur_class):
        for filename in file_generator(dir):
            self.X.append(filename)
            self.y.append(cur_class)

    def create_datasets(self):
        size = len(self.y)
        indices = np.random.permutation(size)
        eighty = int(size*80/100)
        training_idx, test_idx = indices[:eighty], indices[eighty:]
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.training_X, self.test_X = self.X[training_idx], self.X[test_idx]
        self.training_y, self.test_y = self.y[training_idx], self.y[test_idx]


    def generate_training(self, batch_size):
        while True:
            high = len(self.training_X)
            indices = np.random.random_integers(0, high-1, batch_size)
            file_names = self.training_X[indices]
            y = self.training_y[indices]
            ar = np.zeros((batch_size, 150, 150, 3))
            a = 0
            for file_name in file_names:
                #print(file_name, str(y[a]))
                img = cv2.imread(file_name)
                ar[a,:,:,:] = img
                a += 1
            ar = ar.astype('float32')
            ar /= 255
            yield ar, y
                      
    def generate_test(self, batch_size):
        while True:
            high = len(self.test_X)
            indices = np.random.random_integers(0, high-1, batch_size)
            file_names = self.test_X[indices]
            y = self.test_y[indices]
            ar = np.zeros((batch_size, 150, 150, 3))
            a = 0
            for file_name in file_names:
                img = cv2.imread(file_name)
                ar[a,:,:,:] = img
                a += 1
            ar = ar.astype('float32')
            ar /= 255
            yield ar, y
            


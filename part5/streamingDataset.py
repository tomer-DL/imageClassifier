from file_util import file_generator
import numpy as np

class StreamingDataset:
    def __init__(self, dirs, classes):
        self.X = []
        self.y = []
        for i in len(dirs):
            read_dir(dirs[i], classes[i])
        create_datasets()

    def read_dir(self, dir, cur_class):
        for filename in file_generator(dir):
            self.X.append(filename)
            self.y.append(cur_class)

    def create_datasets():
        size = self.y.shape[0]
        indices = np.random.permutation(size)
        eighty = (int)(size*80/100)
        training_idx, test_idx = indices[:eighty], indices[eighty:]
        self.training_X, self.test_X = self.X[training_idx], self.X[test_idx]
        training_y, test_y = self.y[training_idx], self.y[test_idx]


    def generate_training(features, labels, batch_size):
        high = len(self.training_X)
        indices = np.random_integers(0, high, batch_size)
        file_names = self.training_X[indices]
        y = self.y[indices]
        ar = np.zeros((batch_size,features.shape[1], features.shape[2], features.shape[3])
        a = 0
        for file_name in file_names:
            img = cv2.imread(filename)
            ar[a,:,:,:] = img
            a += 1

        return ar, y
                      
            


from variables import *
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
def read_csv_files(csv_file):
        data = pd.read_csv(csv_file).values.astype(np.float32)
        data = shuffle(data)
        Y = data[:,0]
        X = data[:,1:] / 255.0
        X = X.reshape(-1,28, 28, 1)
        return X, Y

def getMnistdata():
    Xtrain, Ytrain = read_csv_files(train_path)
    Xtest , Ytest  = read_csv_files(test_path)
    return Xtrain, Ytrain, Xtest , Ytest

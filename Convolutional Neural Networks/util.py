from variables import *
import pickle
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

def extract_batch_data(file):
        with open(file, 'rb') as data:
                data_dict = pickle.load(data, encoding='bytes')
        return data_dict

def get_train_data():
        train_labels = []
        train_images = []
        for i in range(1, n_batches + 1):
                batch_file = train_batch_prefix + str(i)
                data_dict = extract_batch_data(batch_file)
                train_labels.extend(data_dict[b'labels'])
                train_images.extend(data_dict[b'data'])
        all_train_labels = np.array(train_labels)
        all_train_images = np.array(train_images).reshape(-1,*tensor_shape)
        all_train_labels, all_train_images = shuffle(all_train_labels, all_train_images)
        return all_train_labels, all_train_images


def get_test_data():
        data_dict = extract_batch_data(test_batch)
        test_images = data_dict[b'data'].reshape(-1,*tensor_shape)
        test_labels = data_dict[b'labels']
        all_test_labels, all_test_images = np.array(test_labels), np.array(test_images)
        all_test_labels, all_test_images = shuffle(all_test_labels, all_test_images)
        return all_test_labels, all_test_images

def getCIFAR10data():
        Ytrain, Xtrain = get_train_data()
        Ytest, Xtest = get_test_data()
        return Xtrain, Ytrain, Xtest , Ytest

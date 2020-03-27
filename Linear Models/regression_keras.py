import numpy as np
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from util import get_regression_data

from variables import *

class Regression(object):
    def __init__(self):
        X, Y = get_regression_data()
        self.X = X
        self.Y = Y

    def neural_network(self):
        model = Sequential()
        model.add(Dense(1, input_shape=(1,)))
        model.compile(
            optimizer=SGD(lr,momentum),
            loss='mse')
        model.fit(self.X, self.Y, epochs=num_epochs)
        self.model= model

    # def log_polynomial_regression(self):


if __name__ == "__main__":
    reg = Regression()
    reg.neural_network()

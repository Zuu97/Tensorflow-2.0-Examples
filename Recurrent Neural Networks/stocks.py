import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from matplotlib import pyplot as plt
from util import get_data
from variables import*

import logging
logging.getLogger('tensorflow').disabled = True

class StarBux(object):
    def __init__(self):
        scalar, Xtrain, Ytrain, Xtest, Ytest = get_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.scalar = scalar

    def time_series_model(self):
        inputs = Input(shape=(T, 1)) # N,T,D
        x = LSTM(hidden_dim)(inputs) # N,T,M
        x = Dense(K)(x) # N,T,K

        self.sbux = Model(inputs, x)

    def train_model(self):
        self.time_series_model()
        self.sbux.compile(
            loss='mse',
            optimizer='adam'
        )

        self.history = self.sbux.fit(
                                self.Xtrain,
                                self.Ytrain,
                                epochs=num_epochs,
                                validation_data=[self.Xtest, self.Ytest]
                                )

    def plot_model(self):
        train_loss = self.history.history['loss']
        test_loss = self.history.history['val_loss']

        plt.plot(train_loss, label='Train loss')
        plt.plot(test_loss,  label='Test  loss')
        plt.legend()
        plt.show()

    def predictions(self):
        Ypred = self.sbux.predict(self.Xtest)
        plt.plot(self.Ytest, label='Targets')
        plt.plot(Ypred,  label='Predictions')
        plt.legend()
        plt.savefig('predictions.png')
        plt.show()

if __name__ == "__main__":
    model = StarBux()
    model.train_model()
    model.plot_model()
    model.predictions()
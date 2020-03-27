import numpy as np
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class BreastCancerClassification(object):
    def __init__(self):
        data = load_breast_cancer()
        Xdata = data['data']
        Ydata = data['target']
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, test_size=0.2, random_state=1234, shuffle=True)
        self.scalar = MinMaxScaler()

        Xtrain = self.scalar.fit_transform(Xtrain)
        Xtest = self.scalar.transform(Xtest)

        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest  = Xtest
        self.Ytest  = Ytest

    def classifier(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(30,), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        self.model = model

    def fit(self):
        self.model.compile( optimizer='Rmsprop',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
        self.history =  self.model.fit(
                                self.Xtrain,
                                self.Ytrain,
                                validation_data = [self.Xtest, self.Ytest],
                                epochs=100,
                                batch_size=10
                                )

    def evaluation(self):
        train_loss, train_acc = self.model.evaluate(self.Xtrain, self.Ytrain)
        test_loss , test_acc  = self.model.evaluate(self.Xtest , self.Ytest)
        print("final train loss: ",train_loss, "final train accuracy: ",train_acc)
        print("final test  loss: ",test_loss, "final test  accuracy: ",test_acc)

    def plot_data(self,show_fig=True):
        val_loss = self.history.history['val_loss']
        loss = self.history.history['loss']
        plt.plot(loss, label='loss')
        plt.plot(val_loss, label='val_loss')
        plt.legend()
        plt.show()

        val_acc = self.history.history['val_accuracy']
        acc = self.history.history['accuracy']
        plt.plot(acc, label='acc')
        plt.plot(val_acc, label='val_acc')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    model = BreastCancerClassification()
    model.classifier()
    model.fit()
    model.evaluation()
    model.plot_data(True)
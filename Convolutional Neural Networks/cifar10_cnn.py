import os
from util import getCIFAR10data
import tensorflow as tf
from tensorflow import keras
from keras.models import Model,Input
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from variables import *
import numpy as np
from keras.models import model_from_json

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.995):
            print("\nstop training")
            self.model.stop_training = True


class Cifar10Classifier(object):
    def __init__(self):
        Xtrain, Ytrain, Xtest , Ytest = getCIFAR10data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest


    def mnist_model(self):
        inputs = Input(shape=tensor_shape)
        x = Conv2D(conv1, kernal_size, strides=stride, activation='relu')(inputs)
        x = Conv2D(conv2, kernal_size, strides=stride, activation='relu')(x)
        x = Conv2D(conv3, kernal_size, strides=stride, activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(keep_prob)(x)
        x = Dense(dense, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        x = Dense(output, activation='softmax')(x)

        self.model = Model(inputs, x)

    def train(self):
        callbacks = myCallback()
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
        self.model.summary()
        self.model.fit(
            self.Xtrain,
            self.Ytrain,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=(self.Xtest,self.Ytest),
            callbacks= [callbacks]
            )

    def save_model(self):
        model_json = self.model.to_json()
        with open(saved_model, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(saved_weights)

    def load_model(self):
        json_file = open(saved_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(saved_weights)
        loaded_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
        self.model = loaded_model

    def predict(self,images,labels):
        if not isinstance(images[0], np.ndarray):
            y_pred = np.argmax(self.model.predict(np.array([images]))[0])
            loss, accuracy = self.model.evaluate(images.reshape(1,784),np.array([labels]))
        else:
            y_pred = np.argmax(self.model.predict(images), axis = 1)
            loss, accuracy = self.model.evaluate(images,labels)
        print("Prediction : {}".format(y_pred))
        print("loss : {}".format(loss))
        print("accuracy : {}".format(accuracy))

if __name__ == "__main__":
    Xtrain, Ytrain, Xtest , Ytest = getCIFAR10data()
    classifier = Cifar10Classifier()
    if os.path.exists(saved_weights):
        print("Loading existing model !!!")
        classifier.load_model()
    else:
        print("Training the model  and saving!!!")
        classifier.mnist_model()
        classifier.train()
        classifier.save_model()
    classifier.predict(Xtest,Ytest)
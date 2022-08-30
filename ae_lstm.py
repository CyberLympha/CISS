from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, TimeDistributed, Dropout, RepeatVector

from joblib import dump, load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AE_LSTM:
    def __init__(self, activation="relu", optimizer="adam", loss="mse"):
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.model = Sequential()
        
    def fit(self, x_train, epochs, batch_size, TIME_STEPS):
        n_features = x_train.shape[2]
 
        self.model.add(LSTM(units=64, activation=self.activation, return_sequences=True, 
                       input_shape=(x_train.shape[1], x_train.shape[2])))
        self.model.add(LSTM(units=32, activation=self.activation, return_sequences=False))
        #self.model.add(LSTM(units=16, activation=self.activation, return_sequences=False))

        self.model.add(RepeatVector(TIME_STEPS))

        #self.model.add(LSTM(units=16, activation=self.activation, return_sequences=True))
        self.model.add(LSTM(units=32, activation=self.activation, return_sequences=True))
        self.model.add(LSTM(units=64, activation=self.activation, return_sequences=True))
        self.model.add(TimeDistributed(Dense(n_features)))

        self.model.compile(optimizer=self.optimizer, loss=self.loss)   

        history = self.model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)

        return self
    
    def get_anomalies(self, x_train, x_test, threshold=None, **kwargs):
        test_predict = self.predict(x_test)
        test_predict_error = np.sum(np.mean(abs(x_test - test_predict), axis=1), axis=1)

        anomalies_predict = np.empty_like(test_predict_error)
        
        if threshold != None:
            self.anomaly_threshold = threshold
        else:
            train_predict_error, self.anomaly_threshold = self.get_train_threshold(x_train)
            self.plot_score(train_predict_error, title='train dataframe score')

        self.plot_score(test_predict_error, title='test dataframe score')
        
        np.place(anomalies_predict, test_predict_error <= self.anomaly_threshold, [0])
        np.place(anomalies_predict, test_predict_error > self.anomaly_threshold, [1])
        print(anomalies_predict.shape)

        return anomalies_predict

    def get_train_threshold(self, x_train, print_results=True):
        train_predict = self.predict(x_train)
        predict_error = np.sum(np.mean(abs(x_train - train_predict), axis=1), axis=1)

        if print_results: print(f'''Train prediction score statistics
            min: {predict_error.min()}
            mean: {predict_error.mean()}
            max: {predict_error.max()}''')

        threshold = predict_error.max()
        return predict_error, threshold
    
    def plot_score(self, data, title=''):
        plt.plot(data)
        plt.hlines(self.anomaly_threshold, 0, len(data), color='red')
        plt.title(title)        
        plt.show()
    
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def save_model(self, path):
        self.model.save(f'{path}.h5')

    def load_model(self, path):
        self.model = load(f'{path}.h5')


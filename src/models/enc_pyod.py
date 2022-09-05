import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from pyod.models.auto_encoder import AutoEncoder


class AutoEnc:
    def __init__(self):

        self.model = AutoEncoder(hidden_neurons=[44, 25, 5, 25, 44], epochs=10)

    def fit(self, data, **kwargs):
        self.model.fit(data)

    def predict(self, data):
        return self.model.predict(data)

    def get_anomalies(self, x_train, x_test, threshold=None,  **kwargs):
        test_predict = self.predict(x_test)

        anomalies_predict = np.empty_like(test_predict)

        if threshold != None:
            self.anomaly_threshold = threshold
        else:
            train_predict, self.anomaly_threshold = self.get_train_threshold(x_train)
            self.plot_score(train_predict, title='train dataframe score')

        self.plot_score(test_predict, title='test dataframe score')

        np.place(anomalies_predict, test_predict >= self.anomaly_threshold, [0])
        np.place(anomalies_predict, test_predict < self.anomaly_threshold, [1])

        return anomalies_predict
    

    def get_train_threshold(self, x_train, print_results=True):
        train_predict = self.predict(x_train)

        if print_results: print(f'''Train prediction score statistics
            min: {train_predict.min()}
            mean: {train_predict.mean()}
            max: {train_predict.max()}''')

        threshold = self.model.threshold_

        return train_predict, threshold

    def plot_score(self, data, title=''):
        plt.plot(data)
        plt.hlines(self.anomaly_threshold, 0, len(data), color='red')
        plt.title(title)
        plt.show()

    def save_model(self, path):
        dump(self.model, path)

    def load_model(self, path):
        self.model = load(path)


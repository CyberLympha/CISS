import matplotlib.pyplot as plt
from joblib import dump, load
from pyod.models.auto_encoder import AutoEncoder


class AutoEnc:
    def __init__(self, x_train):
        self.anomaly_threshold = None
        self.model = AutoEncoder(hidden_neurons=[len(x_train), 25, 5, 25, len(x_train)], epochs=5)

    def fit(self, data, **kwargs):
        self.model.fit(data)

    def get_anomalies(self, x_train, x_test, threshold=None, **kwargs):
        anomalies_predict = self.model.predict(x_test)

        if threshold != None:
            self.anomaly_threshold = threshold
        else:
            train_predict, self.anomaly_threshold = self.get_train_threshold(x_train)
            self.plot_score(train_predict, title='train dataframe score')

        self.plot_score(self.model.decision_function(x_test), title='test dataframe score')

        return anomalies_predict

    def get_train_threshold(self, x_train, print_results=True):
        train_predict = self.model.decision_function(x_train)
        threshold = self.model.threshold_
        
        if print_results:
            print(f'''Tthreshold - {self.model.threshold_}''')

        return train_predict, threshold

    def plot_score(self, data, title=''):
        plt.plot(data)
        plt.hlines(self.anomaly_threshold, 0, len(data), color='red')
        plt.title(title)
        plt.show()

    def predict(self, x_test):
        return self.model.predict(x_test)

    def save_model(self, path):
        dump(self.model, path)

    def load_model(self, path):
        self.model = load(path)
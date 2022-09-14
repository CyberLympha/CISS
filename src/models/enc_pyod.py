import matplotlib.pyplot as plt
from joblib import dump, load
from pyod.models.auto_encoder import AutoEncoder


class AutoEnc:
    """
    Auto Encoder (AE) is a type of neural networks for learning useful data
    representations unsupervisedly. Similar to PCA, AE could be used to
    detect outlying objects in the data by calculating the reconstruction
    errors.
    link - https://pyod.readthedocs.io/en/latest/_modules/pyod/models/auto_encoder.html
   """

    def __init__(self, x_train, epochs=None):
        if epochs is None:
            self.epochs = 5
        else:
            self.epochs = epochs
        self.model = AutoEncoder(hidden_neurons=[len(x_train), 25, 5, 25, len(x_train)], epochs=self.epochs)

    def fit(self, data, **kwargs):
        self.model.fit(data)

    def get_anomalies(self, x_train, x_test, threshold=None, **kwargs):
        if threshold != None:
            self.model.threshold_ = threshold

        train_predict = self.model.decision_function(x_train)
        print(f'''Tthreshold - {self.model.threshold_}''')

        self.plot_score(train_predict, title='train dataframe score', flag=False)

        anomalies_predict = self.model.predict(x_test)
        self.plot_score(self.model.decision_function(x_test), title='test dataframe score', flag=True)

        return anomalies_predict

    def plot_score(self, data, title='', flag=None):
        plt.plot(data)
        if flag:
            plt.hlines(self.model.threshold_, 0, len(data), color='red')
        plt.title(title)
        plt.show()

    def predict(self, x_test):
        return self.model.predict(x_test)

    def save_model(self, path):
        dump(self.model, path)

    def load_model(self, path):
        self.model = load(path)

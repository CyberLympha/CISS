import matplotlib.pyplot as plt
from joblib import dump, load
from pyod.models.mo_gaal import MO_GAAL


class MGAAL:
    """
    Multi-Objective Generative Adversarial Active Learning.
    MO_GAAL directly generates informative potential outliers to assist
    the classifier in describing a boundary that can separate outliers
    from normal data effectively.
    link - https://github.com/leibinghe/GAAL-based-outlier-detection
    """

    def __init__(self):
        self.anomaly_threshold = None
        self.model = MO_GAAL(contamination=0.2, stop_epochs=2)

    def fit(self, data, **kwargs):
        self.model.fit(data)

    def get_anomalies(self, x_train, x_test, **kwargs):
        anomalies_predict = self.model.predict(x_test)
        self.plot_score(self.model.decision_function(x_test), title='test dataframe score')

        return anomalies_predict

    def plot_score(self, data, title=''):
        plt.plot(data)
        plt.title(title)
        plt.show()

    def predict(self, x_test):
        return self.model.predict(x_test)

    def save_model(self, path):
        dump(self.model, path)

    def load_model(self, path):
        self.model = load(path)
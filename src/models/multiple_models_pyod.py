import matplotlib.pyplot as plt
from joblib import dump, load
from pyod.models.suod import SUOD


class SU_OD:
    def __init__(self, base_estimators, n_jobs=2, combination='average'):
        self.detector_list = base_estimators
        self.n_jobs = n_jobs
        self.combination = combination

        # decide the number of parallel process, and the combination method
        # then clf can be used as any outlier detection model
        self.model = SUOD(base_estimators=self.detector_list,
                          n_jobs=self.n_jobs,
                          combination=self.combination,
                          verbose=False)

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

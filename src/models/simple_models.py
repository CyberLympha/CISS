import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM 
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV

from joblib import dump, load

class OCSVM:
    def __init__(self, nu=0.00458, kernel="rbf", gamma=0.0008):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma

        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma, verbose=False)
    
    def fit(self, data, **kwargs):
        self.model.fit(data)

    def predict(self, data, **kwargs):
            return self.model.predict(data)

    def get_anomalies(self, x_train, x_test, **kwargs):
        test_predict = self.predict(x_test) 

        anomalies_predict = np.empty_like(test_predict)

        np.place(anomalies_predict, test_predict > 0, [0])
        np.place(anomalies_predict, test_predict < 0, [1])

        return anomalies_predict

    def save_model(self, path):
        dump(self.model, path)

    def load_model(self, path):
        self.model = load(path)


class LOF:
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.model = LocalOutlierFactor(novelty=True, n_jobs=-1, n_neighbors=self.n_neighbors)
    
    def fit(self, data, **kwargs):
        self.model.fit(data)

    def predict(self, data, **kwargs):
        return self.model.score_samples(data)

    def get_anomalies(self, x_train, x_test, threshold=None, **kwargs):
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

        threshold = train_predict.min()
        
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

class iForest:
    def __init__(self, n_estimators=50, contamination=0.001):
        self.n_estimators = n_estimators
        self.contamination = contamination

        self.model = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination, n_jobs=-1)
    
    def fit(self, data, **kwargs):
        self.model.fit(data)

    def predict(self, data, **kwargs):
            return self.model.predict(data)

    def get_anomalies(self, x_train, x_test, **kwargs):
        test_predict = self.predict(x_test) 

        anomalies_predict = np.empty_like(test_predict)

        np.place(anomalies_predict, test_predict > 0, [0])
        np.place(anomalies_predict, test_predict < 0, [1])

        return anomalies_predict

    def get_best_estimator(self, grid_x_train, grid_y_train):
        print('Searching best parameters')

        params = [{'n_estimators': range(60,180,20),
                'contamination': np.arange(0.0, 0.1, 0.05)}]

        grid_search = GridSearchCV(self.model,
                      param_grid=params,
                      scoring='f1',
                      cv=5,
                      verbose=10)

        grid_search.fit(grid_x_train, grid_y_train)

        print(f'Best score: {grid_search.best_score_}')
        print(f'Best params: {grid_search.best_params_}')


        return grid_search.best_estimator_

    def save_model(self, path):
        dump(self.model, path)

    def load_model(self, path):
        self.model = load(path)

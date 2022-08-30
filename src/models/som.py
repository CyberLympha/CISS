from minisom import MiniSom
from joblib import dump, load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance

from matplotlib.gridspec import GridSpec
from ipywidgets import IntProgress
from IPython.display import display

class SOM:
    def __init__(self, x_train, 
                 size=(8,8), 
                 sigma=2., 
                 learning_rate=0.5, 
                 neighborhood_function='gaussian'):
        
        self.x_train = x_train

        self.model = MiniSom(size[0], size[1], 
                             input_len = x_train.shape[1], 
                             sigma=sigma, 
                             learning_rate=learning_rate,
                             neighborhood_function='gaussian')
        
    def fit(self, kostyl, n_iterations=50000, show_winmap=True, **kwarg):
        
        self.model.pca_weights_init(self.x_train)
        print(n_iterations)
        self.model.train_batch(self.x_train, n_iterations, verbose=False)
        
        if show_winmap:        
            win_map = self.model.win_map(self.x_train)

            plt.figure(figsize=(15, 15))
            the_grid = GridSpec(10, 10)
            for position in win_map.keys():
                plt.subplot(the_grid[position[0], position[1]])
                plt.plot(np.min(win_map[position], axis=0), color='gray', alpha=.5)
                plt.plot(np.mean(win_map[position], axis=0))
                plt.plot(np.max(win_map[position], axis=0), color='gray', alpha=.5)
            plt.show()

        return self
    
    def predict(self, x_test):
        neighbourhood_size_X = []
        bar = self.show_bar(size=x_test.shape[0])
    
        for i in x_test: 
            X = i 
            S = self.calc_neighbourhood_size(X, useXvalues = True)
            neighbourhood_size_X.append(S)
            bar.value += 1

            
        return np.array(neighbourhood_size_X)
    
    def get_anomalies(self, x_train, x_test, threshold=None, **kwargs):
        test_predict = self.predict(x_test)
        anomalies_predict = np.empty_like(test_predict)
        
        if threshold != None:
            self.anomaly_threshold = threshold
        else:
            train_predict, self.anomaly_threshold = self.get_train_threshold(x_train)
            self.plot_score(train_predict, title='train dataframe score')

        self.plot_score(test_predict, title='test dataframe score')
        
        np.place(anomalies_predict, test_predict <= self.anomaly_threshold, [0])
        np.place(anomalies_predict, test_predict > self.anomaly_threshold, [1])
        print(anomalies_predict.shape)

        return anomalies_predict

    def get_train_threshold(self, x_train, print_results=True):
        
        neighbourhood_size_X_train = []
        
        bar = self.show_bar(size=x_train.shape[0])
    
        for i in x_train: 
            X = i
            S = self.calc_neighbourhood_size(X, useXvalues = True)
            neighbourhood_size_X_train.append(S)
            bar.value += 1
            
        train_predict = np.array(neighbourhood_size_X_train)

        if print_results: print(f'''Train prediction score statistics
            min: {train_predict.min()}
            mean: {train_predict.mean()}
            max: {train_predict.max()}''')

        threshold = train_predict.max()
        return train_predict, threshold
    
    def calc_SOM_neighbourhood_size(self):
    # подсчет размера окрестности узлов SOM.

        SOM_size = self.model.get_weights().shape[:2]
        dist = np.zeros(SOM_size)

        for row in range(SOM_size[0]):
            for col in range(SOM_size[1]):
                M = SOM.get_weights()[row][col]

                dist_t = distance.cityblock(M,SOM.get_weights()[row-1][col]) if row>0 else 0
                dist_b = distance.cityblock(M,SOM.get_weights()[row+1][col]) if row<SOM_size[0]-1 else 0
                dist_l = distance.cityblock(M,SOM.get_weights()[row][col-1]) if col>0 else 0
                dist_r = distance.cityblock(M,SOM.get_weights()[row][col+1]) if col<col<SOM_size[1]-1 else 0

                dist[row][col] = dist_r + dist_l + dist_t + dist_b

        return dist

    def calc_neighbourhood_size(self, X, useXvalues = False):
    # подсчет размера окрестности узла-победителя

        SOM_size = self.model.get_weights().shape[:2]
        # получаем номер выигравшего узла, его координаты
        M = self.model.winner(X)
        row = M[0]
        col = M[1]

        # если мы хотим вместо весов победителя использовать значения исходного вектора
        if useXvalues:
            M = X
        else:
            M = self.model.get_weights()[row][col]

        dist_t = distance.cityblock(M, self.model.get_weights()[row-1][col]) if row>0 else 0
        dist_b = distance.cityblock(M, self.model.get_weights()[row+1][col]) if row<SOM_size[0]-1 else 0
        dist_l = distance.cityblock(M, self.model.get_weights()[row][col-1]) if col>0 else 0
        dist_r = distance.cityblock(M, self.model.get_weights()[row][col+1]) if col<col<SOM_size[1]-1 else 0


        return dist_r + dist_l + dist_t + dist_b

    def calc_dissimilarity_vectors(self, X, threshold, useXvalues, N=3):
    # расчет векторов dissimirarity
        SOM_size = self.model.get_weights().shape[:2]
        dist = self.calc_SOM_neighbourhood_size(self.model)
        winner_idx = self.model.winner(X)
        dissimilarity = np.zeros(len(X))

        idxs = np.unravel_index(np.argsort(dist, axis=None), dist.shape)
        dissimilarity_vectors = []
        i = 0

        while i<N and i<len(idxs):
            # Если нейрон помечен как аномальный, то пропускаем его
            if dist[idxs[0][i]][idxs[1][i]]>threshold and useXvalues==False:
                N += 1
            else:
                diff = X - self.model.get_weights()[idxs[0][i]][idxs[1][i]]
                dissimilarity_vectors.append(list(diff))
            i += 1

        for i in range(len(dissimilarity_vectors)):
            dissimilarity = dissimilarity + np.array(dissimilarity_vectors[i])
            print("Наиболее вероятные параметры:", np.argsort(dissimilarity)[-3:])

        if len(dissimilarity_vectors) == 0:
            print('There are no healthy nodes')

        return dissimilarity
    
    def plot_score(self, data, title=''):
        plt.plot(data)
        plt.hlines(self.anomaly_threshold, 0, len(data), color='red')
        plt.title(title)        
        plt.show()
    
    def save_model(self, path):
        dump(self.model, f'{path}.joblib')

    def load_model(self, path):
        self.model = load(f'{path}.joblib')

    def show_bar(self, size):
        bar = IntProgress(
            min=0, max=size, 
            description="Computing", style={'bar_color': '#61dc8a'},)
        
        display(bar)
        return bar
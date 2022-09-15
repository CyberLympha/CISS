from math import log, e
import numpy as np
import matplotlib.pyplot as plt
import json

class EntropySensor:
    def __init__(self):
        self.model = {
                'static':set(), 
                'min': None, 
                'mean': None, 
                'max': None
                     }
        
    def fit(self, data, **kwargs):      
        H = []

        for i in range(0, data.shape[0]):
            h = self.get_entropy(data[i])

            H.append(h)
            
            if abs(h) == 0.:
                self.model['static'] |= set([np.mean(data[i])])
        
        self.model['min'] = np.min(H)
        self.model['mean'] = np.mean(H)
        self.model['max'] = np.max(H)
        self.model['static'] = list(self.model['static'])
                
        return self.model
        
    def predict(self, data, **kwargs):
        H = []

        for i in range(0, data.shape[0]):
            h = self.get_entropy(data[i])
            H.append(h)
                    
        return H

    def get_anomalies(self, x_train, x_test, **kwargs):
        train_entropy = self.predict(x_train) 
        self.plot_score(x_train, train_entropy, title='train dataframe score')
        
        test_entropy = self.predict(x_test) 
        self.plot_score(x_test, test_entropy, title='test dataframe score')

        anomalies_predict = []

        for i, value in enumerate(test_entropy):
            
            if abs(value) == 0.:
                if  np.mean(x_test[i]) not in self.model['static']:
                    anomalies_predict.append(1)
                else:
                    anomalies_predict.append(0)
            
            else:
                anomalies_predict.append(0)

        return np.array(anomalies_predict)
    
    def get_entropy(self, data, base=None):
        value,counts = np.unique(data, return_counts=True)

        norm_counts = counts / counts.sum()
        base = e if base is None else base

        return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()
    
    def plot_score(self, data_value, data_H, title=''):   
        plt.plot(data_H)
        plt.hlines(0, 0, len(data_H), color='red', linestyle='--', zorder=10)
        
        for value in self.model['static']:       
            if value != None:        
                static_idxs = np.where(np.mean(data_value, axis=1)==value)[0]               
                if len(static_idxs) > 0:
                    plt.scatter(static_idxs, 
                                [data_H[i] for i in static_idxs], 
                                color='green', s=10)
                    
        plt.title(title)        
        plt.show()

    def save_model(self, path):  
        with open(f'{path}.txt', 'w') as f:
            json.dump(self.model, f)

    def load_model(self, path):
        with open(f'{path}.txt', 'r') as f:
            self.model = json.load(f'{path}.txt')
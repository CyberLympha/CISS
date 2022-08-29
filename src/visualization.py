import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_result(data, y_test, y_pred, descr):
    plt.figure(figsize = (20,10))

    plt.plot(data, alpha=0.5)

    anomaly_idxs = data.iloc[np.where(y_test==1)].index
    anomaly_idxs_pred = data.iloc[np.where(y_pred==1)].index

    plt.plot()
    
    plt.scatter(anomaly_idxs, 
                [2000]*len(anomaly_idxs), c='r', label='anomaly')
    plt.scatter(anomaly_idxs_pred, 
                [2100]*len(anomaly_idxs_pred), c='blue', label=descr)
    
    plt.legend()
    plt.title(f'{descr}. Результаты выявления аномалий')
    plt.show()

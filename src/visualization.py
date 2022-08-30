import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_result(data, y_test, y_pred, descr):
    plt.figure(figsize = (20,10))

    plt.plot(data, alpha=0.5)

    anomaly_idxs = np.where(y_test==1)
    anomaly_idxs_pred = np.where(y_pred==1)
    
    plt.scatter(data.iloc[anomaly_idxs].index, 
                [2000]*len(anomaly_idxs[0]), c='r', label='anomaly')
    plt.scatter(data.iloc[anomaly_idxs_pred].index, 
                [2100]*len(anomaly_idxs_pred[0]), c='blue', label=descr)
    
    plt.legend()
    plt.title(f'{descr}. Результаты выявления аномалий')
    plt.show()

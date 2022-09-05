import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_result(data, y_test, y_pred, descr):
    plt.figure(figsize = (20,10))

    plt.plot(data, alpha=0.5, zorder=0)

    anomaly_idxs = np.where(y_test==1)
    anomaly_idxs_pred = np.where(y_pred==1)
    
    data_max = data.max().max()
    data_min = data.min().min()
    
    anomaly_marker_y = data_min + 0.5*(data_max - data_min)

    
    plt.scatter(data.iloc[anomaly_idxs].index, 
                [anomaly_marker_y]*len(anomaly_idxs[0]), 
                c='r', zorder=10,
                label='anomaly')
    plt.scatter(data.iloc[anomaly_idxs_pred].index, 
                [anomaly_marker_y*0.95]*len(anomaly_idxs_pred[0]), 
                c='blue', zorder=10,
                label=descr)
    
    plt.legend()
    plt.title(f'{descr}. Результаты выявления аномалий')
    plt.show()

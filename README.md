# CISS
Preparation to CISS-2022 competition in Singapour

**SWaT.A1 _ A2_Dec 2015**
| Метод  | Гиперпараметры                                                | Threshold | F1   | Accuracy | Precision | Recall|
| ------ | ------------------------------------------------------------- | --------- | ---- | -------- | --------- | ----- |
| SVM    | nu=0.00458<br>kernel="rbf"<br>gamma=0.0008<br>scaler=Standard | -         | 0.28 | 0.47     | 0.17      | 0.86  |
| SVM    | scaler=MinMax                                                 | -         | 0.26 | 0.37     | 0.15      | 0.89  |
| LOF    | n_components=5<br>n_neighbors=10<br>scaler=Standard           | -165.84   | 0.56 | 0.93     | 1.00      | 0.39  |
| LOF    |                                                               | -30       | 0.73 | 0.94     | 0.91      | 0.60  |
| iForest| n_estimators=50<br>contamination=0.001<br>scaler=Standard     | -         | 0.72 | 0.95     | 1.00      | 0.57  |
| iForest| n_estimators=200<br>contamination=0.005                       | -         | 0.74 | 0.95     | 0.98      | 0.59  |
| LSTM-AE| 64-32-32-64<br>window_size=30<br>resampling=1m                | 5977      | 0.00 | 0.88     | 0.00      | 0.00  |
| LSTM-AE|                                                               | 800       | 0.00 | 0.88     | 0.60      | 0.00  |
| LSTM-AE|                                                               | 400       | 0.74 | 0.95     | 0.98      | 0.59  |
| SOM    | 8x8<br>sigma=2.0<br>learning_rate=0.5                         | 784       | 0.00 | 0.88     | 0.00      | 0.00  |
| SOM    |                                                               | 220       | 0.74 | 0.95     | 0.98      | 0.59  |

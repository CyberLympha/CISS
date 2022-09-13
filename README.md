# CISS
Preparation to CISS-2022 competition in Singapour

# Результаты работы моделей

**SWaT.A1 _ A2_Dec 2015**
| Метод  | Гиперпараметры                                                | Threshold | F1   | Accuracy | Precision | Recall|
| ------ | ------------------------------------------------------------- | --------- | ---- | -------- | --------- | ----- |
| SVM    | <sub>nu=0.00458<br>kernel="rbf"<br>gamma=0.0008<br>scaler=Standard | -         | 0.28 | 0.47     | 0.17      | 0.86  |
| SVM    | <sub>scaler=MinMax                                                 | -         | 0.26 | 0.37     | 0.15      | 0.89  |
| LOF    | <sub>n_components=5<br>n_neighbors=10<br>scaler=Standard           | -165.84   | 0.56 | 0.93     | 1.00      | 0.39  |
| LOF    | <sub>                                                              | -30       | 0.73 | 0.94     | 0.91      | 0.60  |
| iForest| <sub>n_estimators=50<br>contamination=0.001<br>scaler=Standard     | -         | 0.72 | 0.95     | 1.00      | 0.57  |
| iForest| <sub>n_estimators=200<br>contamination=0.005                       | -         | 0.74 | 0.95     | 0.98      | 0.59  |
| LSTM-AE| <sub>64-32-32-64<br>window_size=30<br>resampling=1m<br>scaler=Standard | 5977      | 0.00 | 0.88     | 0.00      | 0.00  |
| LSTM-AE| <sub>                                                              | 800       | 0.00 | 0.88     | 0.60      | 0.00  |
| LSTM-AE| <sub>                                                              | 400       | 0.74 | 0.95     | 1.00      | 0.59  |
| MLP-AE | <sub> 16-8-4-4-8-16                                                |           | 0.74 | 0.95     | 1.00      | 0.59  |
| SOM    | <sub>8x8<br>sigma=2.0<br>learning_rate=0.5                         | 784       | 0.00 | 0.88     | 0.00      | 0.00  |
| SOM    | <sub>                                                              | 220       | 0.74 | 0.95     | 0.91      | 0.65  |

**SWaT.A4 _ A5_Jul 2019**
| Метод  | Гиперпараметры                                                | Threshold | F1   | Accuracy | Precision | Recall|
| ------ | ------------------------------------------------------------- | --------- | ---- | -------- | --------- | ----- |
| SVM    | <sub>nu=0.00458<br>kernel="rbf"<br>gamma=0.0008<br>scaler=Standard | -         | 0.49 | 0.50     | 0.35      | 0.81  |
| SVM    | <sub>scaler=MinMax                                                 | -         | 0.54 | 0.52     | 0.38      | 0.93  |
| LOF    | <sub>n_components=5<br>n_neighbors=10<br>scaler=Standard           | -4.5      | 0.49 | 0.62     | 0.41      | 0.60  |
| iForest| <sub>n_estimators=50<br>contamination=0.001<br>scaler=Standard     | -         | 0.00 | 0.70     | 0.00      | 0.00  |
| iForest| <sub>n_estimators=200<br>contamination=0.05                        | -         | 0.56 | 0.65     | 0.45      | 0.74  |
| LSTM-AE| <sub>64-32-32-64<br>window_size=30<br>resampling=1m<br>scaler=Standard | 200   | 0.49 | 0.49     | 0.35      | 0.81  |
| LSTM-AE| <sub>                                                              | 80.5      | 0.48 | 0.49     | 0.35      | 0.78  |
| MLP-AE | <sub> 16-8-4-4-8-16                                                | 1.89      | 0.45 | 0.63     | 0.40      | 0.52  |
| MLP-AE | <sub>                                                              | 1.5       | 0.51 | 0.55     | 0.38      | 0.79  |
| SOM    | <sub>8x8<br>sigma=2.0<br>learning_rate=0.5                         | 142       | 0.36 | 0.67     | 0.44      | 0.31  |
| SOM    | <sub>                                                              | 100       | 0.37 | 0.62     | 0.37      | 0.37  | 
  
 **WADI.A2_19 Nov 2019**
| Метод  | Гиперпараметры                                                | Threshold | F1   | Accuracy | Precision | Recall|
| ------ | ------------------------------------------------------------- | --------- | ---- | -------- | --------- | ----- |
| SVM    | <sub>nu=0.00458<br>kernel="rbf"<br>gamma=0.0008<br>scaler=Standard | -         | 0.10 | 0.45     | 0.05      | 0.51  |
| SVM    | <sub>без "плохих" признаков                                        | -         | 0.39 | 0.95     | 0.71      | 0.27  |
| SVM    | <sub>scaler=MinMax                                                 | -         | 0.48 | 0.95     | 0.58      | 0.42  |
| LOF    | <sub>n_components=5<br>n_neighbors=10<br>scaler=Standard           | -89,3     | 0.00 | 0.94     | 1.00      | 0.00  |
| LOF    | <sub>                                                              | -20       | 0.33 | 0.94     | 0.52      | 0.24  |
| iForest| <sub>n_estimators=50<br>contamination=0.001<br>scaler=Standard     | -         | 0.00 | 0.94     | 0.03      | 0.00  |
| iForest| <sub>n_estimators=98<br>contamination=0.04                         | -         | 0.28 | 0.88     | 0.21      | 0.41  |
| LSTM-AE| <sub>64-32-32-64<br>window_size=10<br>resampling=1m<br>scaler=Standard | 1865      | 0.08 | 0.44     | 0.04      | 0.39  |
| LSTM-AE| <sub>без "плохих" признаков                                        | 180       | 0.31 | 0.95     | 0.90      | 0.19  |
| LSTM-AE| <sub>scaler=MinMax                                                 | 14        | 0.45 | 0.95     | 0.68      | 0.34  |
| MLP-AE | <sub> 16-8-4-4-8-16                                                | 2.5       | 0.08 | 0.44     | 0.04      | 0.39  |
| MLP-AE | <sub> без "плохих признаков"                                       | 2.15      | 0.11 | 0.95     | 0.99      | 0.06  |
| MLP-AE | <sub>                                                              | 1.4       | 0.30 | 0.95     | 0.81      | 0.18  |
| SOM    | <sub>8x8<br>sigma=2.0<br>learning_rate=0.5                         | 800       | 0.00 | 0.94     | 0.00      | 0.00  |
| SOM    | <sub>                                                              | 400       | 0.18 | 0.94     | 0.49      | 0.11  |

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter

def prepare_data(data_train, data_test):
    
    # deleting NaN features
    nan_col = data_train.isna().sum()[data_train.isna().sum()==data_train.shape[0]].index
    
    print(f'deleting NaN features: {nan_col}\n')
    df_train = data_train.drop(nan_col, axis=1)
    df_test = data_test.drop(nan_col, axis=1)
   
    # deleting constant features
    const_col_train, const_col_test = [], []

    for col in df_train.columns:
        if len(df_train[col].unique()) == 1:
            const_col_train.append(col)

    for col in df_test.columns:
        if len(df_test[col].unique()) == 1:
            const_col_test.append(col)
            
    const_col = []

    for col in const_col_test:
        if col in const_col_train:
            const_col.append(col)
            
    print(f'deleting constant features: {const_col}')
    df_train = df_train.drop(const_col, axis=1).dropna()
    df_test = df_test.drop(const_col, axis=1).dropna()

    return df_train, df_test

def create_sequences(values, time_steps):

    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])

    return np.stack(output)

def get_traintest(df_train, df_test, 
                  label_dict = None,
                  reshape=False, resample_rate = None,
                  window_size=None, scaler='Standard'):
    
    print(f'Scaling... ({scaler})')
    if scaler == 'Standard':
        scaler = StandardScaler()
    elif scaler == 'MinMax':
        scaler = MinMaxScaler()

    scaler.fit(df_train)
    
    df_label = df_test.pop('anomaly')
    if label_dict != None:    
        y_test = np.array([label_dict[x] for x in df_label.values])
    
    else:
        y_test = np.array(df_label.values)

    
    x_train = scaler.transform(df_train.dropna())
    x_test = scaler.transform(df_test)
        
    print(f'Количество аномалий: {(sum(y_test) / y_test.shape[0] * 100):.1f}%\n')
    print(Counter(y_test))

    if resample_rate != None:
        print(f'Resampling... ({resample_rate})')

        df_train = df_train.resample(resample_rate).mean().fillna(method='ffill') 
        df_test = df_test.resample(resample_rate).mean().fillna(method='ffill') 
        df_label = df_label.resample(resample_rate).mean().fillna(method='ffill') 

        print(f'{df_train.shape}, {df_test.shape}, {df_label.shape},')

        # df_train = df_train.resample(resample_rate).mean().fillna(method='ffill')
        # df_label = df_label.resample(resample_rate).median().fillna(method='ffill')

    if reshape:
        print(f'Reshaping...')

        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    if window_size != None:
        print(f'Create sequences with window size {window_size}...')

        x_train = create_sequences(x_train, window_size)
        x_test = create_sequences(x_test, window_size)
        y_test = create_sequences(y_test, window_size)
        
    print(f'''Размеры выборок:
            x_train: {x_train.shape}
            x_test: {x_test.shape}
            y_test: {y_test.shape}''')

    return x_train, x_test, y_test
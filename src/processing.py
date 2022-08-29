import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime

import sklearn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

MODEL_STORE_PATH = './saved_models/'

class Predictor:
    
    def __init__(self, model, data: list, descr: str, resave_model=False, window_size=None):
        self.model = model
        self.descr = descr
        self.resave_model = resave_model

        self.x_train = data[0]
        self.x_test = data[1]
        self.label = data[2]
        self.window_size = window_size

        self.load_model()
        
    def load_model(self):
        model_file = f'{MODEL_STORE_PATH}{self.descr}.joblib'
                   
        if self.resave_model or not os.path.exists(model_file):

            if self.resave_model: 
                print(f"{str(datetime.now())}: refit model...")
            elif not os.path.exists(model_file): 
                print(f"{str(datetime.now())}: can't find saved model, fit model...")
            
            self.model.fit(self.x_train, 
                           epochs=10, 
                           batch_size=1000, 
                           TIME_STEPS=self.window_size)

            self.model.save_model(model_file)    

        else:
            print(f"{str(datetime.now())}: find saved model: {model_file}, loading...")
            self.model.load_model(model_file) 
            
    def tuning_model(self):
        model_file = f'{MODEL_STORE_PATH}{self.descr}_tuned.joblib'
        self.model = self.model.get_best_estimator(self.x_test, self.label)
        self.model.fit(self.x_train)
        self.model.save_model(model_file) 

    
    def get_anomalies(self, threshold=None):
        self.label_predict = self.model.get_anomalies(self.x_train, self.x_test, threshold=threshold)

        return self.label_predict
           
    def get_score(self, print_results=True):

        if self.window_size == None:
            y_test = self.label
            y_pred = self.label_predict

        else:
            y_test = self.label[:,0]
            y_pred = self.label_predict

        f1_score_ = f1_score(y_test, y_pred)
        accuracy_score_ = accuracy_score(y_test, y_pred)
        precision_score_ = precision_score(y_test, y_pred)
        recall_score_ = recall_score(y_test, y_pred)

        if print_results:
            print(self.descr)
            print(f'f1_score: {f1_score_:0.2f}')
            print(f'accuracy_score: {accuracy_score_:0.2f}')
            print(f'precision_score: {precision_score_:0.2f}')
            print(f'recall_score: {recall_score_:0.2f}')   

        return {'f1_score':f1_score_,
                'accuracy_score':accuracy_score_,
                'precision_score':precision_score_,
                'recall_score':recall_score_,}
            
    def _show_bar(self, size):
        bar = IntProgress(
            min=0, max=len(size), 
            description="Computing", style={'bar_color': '#61dc8a'},)
        
        display(bar)
        return bar
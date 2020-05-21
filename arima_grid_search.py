#!/usr/bin/env python
# coding: utf-8

# This notebook grid searchs for the best model for ARIMA based on the dataset.

# In[8]:


# import libraries

# ARIMA libraries
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
import warnings

# Regular libraries
import numpy as np
import pandas as pd


# In[9]:


def evaluate_arima_model(X, arima_order):
    """evaluate an ARIMA model for a given order (p,d,q) and return RMSE"""    
    
    # prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    
    # make predictions
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        
        # fit model
        try:
            model_fit = model.fit(trend='nc', disp=0)
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        except:
            continue
        
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    
    return rmse


# In[10]:


def evaluate_models(dataset, p_values, d_values, q_values):
    """evaluate combinations of p, d and q values for an ARIMA model"""    
    
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
                   
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    
    return best_cfg


# In[11]:


def driver(df, arg_dict):
    """Driver program for finding best_cfg for ARIMA"""
    # load dataset
    series = df[arg_dict['dependent_variable']]

    # evaluate parameters
    p_values = arg_dict['p_values']
    d_values = arg_dict['d_values']
    q_values = arg_dict['q_values']
    warnings.filterwarnings("ignore")
    best_cfg = evaluate_models(series.values, p_values, d_values, q_values)
    
    return best_cfg


# In[12]:


if __name__ == '__main__':
    
    
    # Arguments for driver
    arg_dict = {'file_name_1': r'data\all_df.csv',
                'file_name_2': r'C:\Users\linds\OneDrive\mystuff\GitHub\covid\data\country_codes_edited.csv',
                'feature': 'Province_State',
                'place': 'New York',
                'dependent_variable': 'Deaths',
                'path': r'C:\Users\linds\OneDrive\mystuff\GitHub\COVID-19\csse_covid_19_data\csse_covid_19_daily_reports',
                'p_values': range(0,3),
                'd_values': range(0,3),
                'q_values': range(0,3),
                'split_value': .5,
                'bias': 0,
                'date': '12-31-2020'}
    
    df = pd.read_csv('df.csv')
    
    # Start driver
    best_cfg = driver(df, arg_dict)
    


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# This notebook grid searchs for the best model for ARIMA based on the dataset.

# In[193]:


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


# In[194]:


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


# In[195]:


def evaluate_models(dataset, p_values, d_values, q_values):
    """evaluate combinations of p, d and q values for an ARIMA model"""    
    
    dataset = dataset.astype('float32')
    rmse_list = []

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    rmse_list.append((rmse, order))
                    print('RMSE is', round(rmse,3), 'with ARIMA of', order)
                except:
                    continue
    
    # Sort the RMSEs
    rmse_list.sort(key=lambda tup: tup[0])
    
    print('\nBest RMSE Score is {} with ARIMA of {}'.format(round(rmse_list[0][0],3), 
                                                                      rmse_list[0][1]))
    
    # We just need the order, not the RMSE
    order_list = [item[1] for item in rmse_list]

    return order_list


# In[196]:


def driver(df, arg_dict):
    """Driver program for finding best_cfg for ARIMA. Returned as a sorted list of all results"""
    # load dataset
    series = df[arg_dict['dependent_variable']]

    # evaluate parameters
    p_values = arg_dict['p_values']
    d_values = arg_dict['d_values']
    q_values = arg_dict['q_values']
    warnings.filterwarnings("ignore")
    order_list = evaluate_models(series.values, p_values, d_values, q_values)
    
    return order_list


# In[197]:


if __name__ == '__main__':
    
    
    # Arguments for driver
    arg_dict = {'file_name_1': r'data\all_df.csv',
                'file_name_2': r'C:\Users\linds\OneDrive\mystuff\GitHub\covid\data\country_codes_edited.csv',
                'feature': 'Alpha_3',
                'place': 'USA',
                'dependent_variable': 'Confirmed',
                'path': r'C:\Users\linds\OneDrive\mystuff\GitHub\COVID-19\csse_covid_19_data\csse_covid_19_daily_reports',
                'p_values': range(0,2),
                'd_values': range(0,2),
                'q_values': range(0,2),
                'split_value': .5,
                'bias': 0,
                'date': '12-31-2020'}
    
    df = pd.read_csv('df.csv')
    
    # Start driver
    order_list = driver(df, arg_dict)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# This notebook grid searchs for the best model for ARIMA based on the dataset.

# In[1]:


# import libraries

# ARIMA libraries
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults

# Parallel Libraries
from multiprocessing import cpu_count
from joblib import Parallel, delayed

# Regular libraries
import itertools
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[18]:


def evaluate_arima_model(args):
    """evaluate an ARIMA model for a given order (p,d,q) and return RMSE and order"""
    
    # distribute args to appropriate variables
    test, history, order = args
    
    # make predictions
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=order)
        
        # fit model
        try:
            model_fit = model.fit(trend='nc', disp=0)
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        except:
            continue
        
    # calculate out of sample error
    try:
        rmse = sqrt(mean_squared_error(test, predictions))
        print('RMSE is', round(rmse,3), 'with ARIMA of', order)
    
    except Exception as e:
        print(e)
        print('Model did not fit/predict so unable to compute RMSE for order', order)
        rmse = 999999
    
    return rmse, order
    


# In[19]:


def evaluate_models(X, arima_list):
    """evaluate combinations of p, d and q values for an ARIMA model"""    
    
    # prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = list(train)
    rmse_list = []
    
    # Need to create the same number of inputs for each argument into the parallel function
    test_list = len(arima_list)*[X]
    history_list = len(arima_list)*[history]
    zip_list = list(zip(test_list, history_list, arima_list))
    
    # call function and run in parallel
    rmse_list = Parallel(n_jobs=-1, verbose=10)(delayed(evaluate_arima_model)(args) for args in zip_list)
    
    # Sort the RMSEs
    rmse_list.sort(key=lambda tup: tup[0])
    
    # Sometimes, we do not have any ARIMA models that successfully fit and predict.
    try:
        print(f'\nBest RMSE Score is {round(rmse_list[0][0],3)} with ARIMA of {rmse_list[0][1]}')
    except:
        print('No ARIMA models fit and predicted successfully. Try different p,d,q parameters')

    return rmse_list


# In[20]:


def driver(df, arg_dict):
    """Driver program for finding best_cfg for ARIMA. Returned as a sorted list of all results"""
    # load dataset
    series = df[arg_dict['dependent_variable']]

    # evaluate parameters
    p_values = arg_dict['p_values']
    d_values = arg_dict['d_values']
    q_values = arg_dict['q_values']
    
    # Generate all different combinations of p, d and q triplets
    arima_list = list(itertools.product(p_values, d_values, q_values))
    
    # Grid search the possibilities
    rmse_list = evaluate_models(series.values, arima_list)
    
    return rmse_list


# In[21]:


if __name__ == '__main__':
    
    
    # Arguments for driver
    arg_dict = {'file_name_1': r'data\all_df.csv',
                'file_name_2': r'C:\Users\linds\OneDrive\mystuff\GitHub\covid\data\country_codes_edited.csv',
                'feature': 'Alpha_3',
                'place': 'USA',
                'dependent_variable': 'Deaths',
                'path': r'C:\Users\linds\OneDrive\mystuff\GitHub\COVID-19\csse_covid_19_data\csse_covid_19_daily_reports',
                'p_values': range(0,2),
                'd_values': range(0,2),
                'q_values': range(0,2),
                'split_value': .5,
                'bias': 0,
                'date': '12-31-2020'}
    
    df = pd.read_csv('df.csv')
    
    # Start driver
    rmse_list = driver(df, arg_dict)
    
    # Create a df to display the rmse and pdq
    rmse_order_df = pd.DataFrame({'RMSE': [x[0] for x in rmse_list], 'Order': [x[1] for x in rmse_list]})
    print(rmse_order_df)


# In[ ]:





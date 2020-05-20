#!/usr/bin/env python
# coding: utf-8

# Multi Step Forecast with ARIMA.

# In[26]:


# import libraries
from datetime import datetime, timedelta
import dateutil.parser
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (12,8)
import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.arima_model import ARIMA


# In[27]:


def place_value(number): 
    """Changes number to a readable number with ','s for 000s'"""
    
    number = int(number)
    
    return ("{:,}".format(number))


# In[28]:


def difference(dataset, interval=1):
    """create a differenced series"""
    
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
        
    return np.array(diff)


# In[29]:


def inverse_difference(history, yhat, interval=1):
    """invert differenced value"""
    
    return yhat + history[-interval]


# In[33]:


def forecast_multi_step(df, arg_dict):
    """Create a forecast multiple time periods into the future"""
    
    # load dataset
    series = df[arg_dict['dependent_variable']]

    # This algorithm accommodates seasonal variation which may be helpful if there is a season to covid.
    # Flu season runs from November thru March with a peak in December and February each year.
    # This is for the northern hemisphere. Obviously the opposite for the southern hemisphere.
    # To get the maximum information into the model currently use length /2.
    X = series.values
    period = int(X.shape[0]/2)
    differenced = difference(X, period)
    
    # The above may introduce some NaNs
    differenced = differenced[~np.isnan(differenced)]

    # fit model
    model = ARIMA(differenced, order=(arg_dict['best_cfg'])) 
    model_fit = model.fit(disp=0)

    # Create time period to report on
    start = df.index[-1]
    end = pd.to_datetime(arg_dict['date'], infer_datetime_format=True)
    steps = end - start
    steps = (steps.days) + 1 # Add 1 day

    # Fit the out of sample
    forecast = model_fit.forecast(steps=steps)[0]

    # invert the differenced forecast to something usable
    history = [x for x in X]
    inverted_ = []

    for day, yhat in enumerate(forecast):

        inverted = inverse_difference(history, yhat, period)
        history.append(inverted)
        inverted_.append(int(inverted))

    # Create a df for reporting
    # Create a date_range index
    start = df.index[-1]
    end = start + timedelta(days=steps-1)

    # Create the df
    forecast_df = pd.DataFrame({arg_dict['dependent_variable']: inverted_}, index=pd.date_range(start=start, end=end))

    # Shift Deaths by one day to make it lineup correctly with the date. 
    forecast_df[arg_dict['dependent_variable']] = forecast_df[arg_dict['dependent_variable']].shift(1)
    forecast_df = forecast_df[1:]
    
    return forecast_df


# In[34]:


def forecast(forecast_df, arg_dict):
    """Print out the prediction in a readable format"""
    
    predicted = place_value(int(forecast_df[arg_dict['dependent_variable']].iloc[-1]))
    forecast_date = forecast_df.index[-1].strftime('%Y-%m-%d')
    print(f'The {arg_dict["place"]} prediction is for {predicted} cumulative {arg_dict["dependent_variable"]} to occur by {forecast_date}')


# In[38]:


def plot_multi_step_forecast(forecast_df, arg_dict):
    """Plot the multi step forecast"""
    
    # Add a new column that is dependent_variable per million
    forecast_df[arg_dict['dependent_variable'] + '_e6'] = forecast_df[arg_dict['dependent_variable']] / 1000000

    # Assemble title
    start = forecast_df.index[0].strftime('%Y-%m-%d')
    title = ('Forecast Cumulative {} for {} In Millions For Covid-19 ({} to {})').format(
        arg_dict['dependent_variable'], arg_dict['place'], start, arg_dict['date'])
    plt.title(title)

    # Create x and y axis labels
    plt.xlabel('Date')
    ylabel_ = ('Cumulative {} In Millions').format(arg_dict['dependent_variable'])
    plt.ylabel(ylabel_)

    # Create plot
    plt.plot(arg_dict['dependent_variable'] + '_e6', data=forecast_df, linewidth=4, 
             label=arg_dict['dependent_variable'] + '_e6')
    plt.legend()
    plt.savefig(r'pics/' + arg_dict['place'] + '_prediction.png');
    


# In[39]:


def driver(df, arg_dict):
    """driver function for plot, save, forecast"""
    
    # Run forecast into the future
    forecast_df = forecast_multi_step(df, arg_dict)
    
    # Report on results
    forecast(forecast_df, arg_dict)
    
    # Plot results
    plot_multi_step_forecast(forecast_df, arg_dict)
    
    return forecast_df


# In[40]:


if __name__ == '__main__':
    
    # Prepare arguments for driver
    with open('arg_dict.pickle', 'rb') as handle:
        arg_dict = pickle.load(handle)
    
    arg_dict.update({'date': '12-31-2020'})
    df = pd.read_csv('df.csv', parse_dates=True, index_col='Date_')
    
    # Start driver
    forecast_df = driver(df, arg_dict) 
    


# In[ ]:





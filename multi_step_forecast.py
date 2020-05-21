#!/usr/bin/env python
# coding: utf-8

# Multi Step Forecast with ARIMA.

# In[178]:


# import libraries
from datetime import datetime, timedelta
import dateutil.parser
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (12,8)
import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.arima_model import ARIMA


# In[179]:


def place_value(number): 
    """Changes number to a readable number with ','s for 000s'"""
    
    number = int(number)
    
    return ("{:,}".format(number))


# In[180]:


def difference(dataset, interval=1):
    """create a differenced series"""
    
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
        
    return np.array(diff)


# In[181]:


def inverse_difference(arg_dict, history, yhat, interval=1):
    """invert differenced value"""
    
    # Include bias from previous optimization runs
    return yhat + arg_dict['bias'] + history[-interval]


# In[182]:


def forecast_multi_step(df, arg_dict):
    """Create a forecast multiple time periods into the future"""
    
    # load dataset
    series = df[arg_dict['dependent_variable']]

    # This algorithm accommodates seasonal variation which may be helpful if there is a season to covid.
    # Flu season runs from November thru March with a peak in December and February each year.
    # This is for the northern hemisphere. Obviously the opposite for the southern hemisphere.
    # To simply ignore seasonality set period to 1
    X = series.values
    #period = int(X.shape[0]/2)
    period = 1
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

        inverted = inverse_difference(arg_dict, history, yhat, period)
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


# In[183]:


def forecast(forecast_df, arg_dict):
    """Print out the prediction in a readable format"""
    
    # Some countries are flattening the curve. As a result ARIMA will forecast fewer 'Deaths'.
    # Then the cumulative numbers start decreasing and will go negative. Lets catch that here.
    
    # Just want a normal integer index
    forecast_df.reset_index(inplace=True)
    forecast_df.rename(columns={'index': 'Date_'}, inplace=True)

    # Get max values
    max_forecast = forecast_df[arg_dict['dependent_variable']].max()
    idx_max = forecast_df[arg_dict['dependent_variable']].idxmax()
    last_forecast = forecast_df[arg_dict['dependent_variable']].iloc[-1]

    if last_forecast < max_forecast:
        # Truncate forecast_df
        forecast_df = forecast_df.loc[:idx_max+1]
    else:
        pass

    predicted = place_value(int(forecast_df[arg_dict['dependent_variable']].iloc[-1]))
    forecast_date = forecast_df['Date_'].iloc[-1]
    print(f'The {arg_dict["place"]} prediction is for {predicted} cumulative {arg_dict["dependent_variable"]} to occur by {forecast_date}')

    # Make Date_ the index again for plotting purposes
    forecast_df.set_index('Date_', drop='Date_', inplace=True)

    return forecast_df
          


# In[184]:


def plot_multi_step_forecast(forecast_df, arg_dict):
    """Plot the multi step forecast"""
    
    # Add a new column that is dependent_variable per million
    forecast_df[arg_dict['dependent_variable'] + '_e6'] = forecast_df[arg_dict['dependent_variable']] / 1000000

    # Assemble title
    start = forecast_df.index[0].strftime('%Y-%m-%d')
    end = forecast_df.index[-1].strftime('%Y-%m-%d')
    title = ('Forecast Cumulative {} for {} In Millions From {} to {}').format(
        arg_dict['dependent_variable'], arg_dict['place'], start, end)
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
    


# In[185]:


def driver(df, arg_dict):
    """driver function for plot, save, forecast"""
    
    # Run forecast into the future
    forecast_df = forecast_multi_step(df, arg_dict)
    
    # Report on results
    forecast_df = forecast(forecast_df, arg_dict)
    
    # Plot results
    plot_multi_step_forecast(forecast_df, arg_dict)
    
    return forecast_df


# In[186]:


if __name__ == '__main__':
    
    # Prepare arguments for driver
    with open('arg_dict.pickle', 'rb') as handle:
        arg_dict = pickle.load(handle)
    
    df = pd.read_csv('df.csv', parse_dates=True, index_col='Date_')
    
    # Start driver
    forecast_df = driver(df, arg_dict) 
    


# In[ ]:





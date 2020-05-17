#!/usr/bin/env python
# coding: utf-8

# This code loads the data from the John Hopkins dataset. The best way is to make sure that you have the current version of the data by cloning the repository from John Hokins at this url https://github.com/CSSEGISandData/COVID-19

# In[47]:


# import required libraries
import glob
import numpy as np
import pandas as pd


# In[48]:


def get_all_columns(arg_dict):
    """Columns have changed over time. Need to get the largest possible number of column names"""
    
    all_files = glob.glob(arg_dict['path'] + "/*.csv")

    li_set = {}

    for filename in all_files:
        df = pd.read_csv(filename, nrows=1) # Just want the header
        cols = df.columns
        cols = set(cols)
        li_set = cols.union(li_set)

    return li_set, df.columns


# In[49]:


def get_data(all_df_cols, arg_dict):
    """Gets the data from the John Hopkins directory which is stored locally"""

    all_files = glob.glob(arg_dict['path'] + "/*.csv")

    li = []

    for filename in all_files:

        # Read in the next csv
        df = pd.read_csv(filename)

        # Get the date of the file
        date_string = filename[-14:-4]

        # Insert the date_string into the first column of the df
        df.insert(0, "Date_", date_string) 

        # If it is a full df then just append it
        if df.shape[1] == 13:
            li.append(df)

        else:
            # Need to build a df that is the same number of columns as the largest df (last one)
            # Create a temp_df to hold everything
            temp_df = pd.DataFrame(data=np.nan, columns=all_df_cols.insert(0, 'Date_'), index=df.index)

            # Call function to create the short_df
            df = short_df(df, temp_df)

            # append df to li
            li.append(df)

    return pd.concat(li)


# In[50]:


def short_df(df, temp_df):
    """Takes the columns from the 'short df' and puts them in a temp df.
    This temp_df will later be put into a df with other columns."""
    
    # Pandas really does not like '/' or ' ' in a column name so ... time to rename
    df.rename(columns={'Province/State': 'Province_State', 
                       'Country/Region': 'Country_Region',
                       'Last Update': 'Last_Update',
                       'Latitude': 'Lat',
                       'Longitude': 'Long_'}, inplace=True)
    
    # Get the current_df columns
    cols = df.columns
    
    # For loop for putting the appropriate columns in temp_df
    for col in cols:
        temp_df[col] = df[col]
        
    return temp_df


# In[51]:


def fix_date_index_write(all_df, arg_dict):
    """Resets the index, changes Data_ to datetime format, and writes all_df to disk"""
    
    # reset the index
    all_df.reset_index(inplace=True, drop=True)

    # Convert Date_ column to date
    all_df['Date_'] =  pd.to_datetime(all_df['Date_'], infer_datetime_format=True)
    
    # Write to disk
    all_df.to_csv(arg_dict['file_name_1'])
    
    return all_df


# In[52]:


def driver(arg_dict):
    
    li_set, all_df_cols = get_all_columns(arg_dict)
    df = get_data(all_df_cols, arg_dict)
    df = fix_date_index_write(df, arg_dict)
    
    return df, li_set


# In[53]:


if __name__ == '__main__':
    
    # Prepare arguments for driver
    arg_dict = {'file_name_1': r'data\all_df.csv',
                'file_name_2': r'C:\Users\linds\OneDrive\mystuff\GitHub\covid\data\country_codes_edited.csv',
                'feature': 'Alpha_3',
                'place': 'USA',
                'dependent_variable': 'Deaths',
                'path': r'C:\Users\linds\OneDrive\mystuff\GitHub\COVID-19\csse_covid_19_data\csse_covid_19_daily_reports'}
    
    # Start driver
    df, li_set = driver(arg_dict) # Want the list of column names in case they change
    


# In[ ]:





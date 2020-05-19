# arima
Auto Regressive Integrated Moving Average
May 18, 2020


## Table of Contents

- [Installation](#installation)
- [Project Motivation](#motivation)
- [File Descriptions](#files)
- [Results](#results)
- [Deploy](#deploy)
- [Comments](#comments)
- [Licensing, Authors, and Acknowledgements](#licensing)


## Installation <a name="installation"></a>

The code runs with Python version 3.6.3. There are a number of libraries required to run this notebook. Those are for ARIMA and "regular libraries". Please go to the # import libraries cell which is near the top of the notebook and .py files for the definitive list.


## Project Motivation<a name="motivation"></a>

It can be difficult to get accurate  and time sensitive information on how countries are doing at present during the SARS-2 pandemic (COVID-19 is the illness, the pandemic is the virus).  This repository allows one to quickly answer forecasting questions regarding cases and deaths for the large variety of jurisdictions that are in the John Hopkins GitHub repository (below).


## File Descriptions <a name="files"></a>

The following data files (available in this repository) are required.

- I am using data from John Hopkins University. There is a GitHub repository that holds this data. To refresh it just go to https://github.com/CSSEGISandData/COVID-19 and hit the Clone of download button. It will ask to launch GitHub Desktop and will update it automatically.

- World Bank Country Codes
https://wits.worldbank.org/wits/wits/witshelp/content/codes/country_codes.htm


## Results<a name="results"></a>

This repository allows you to quickly make predictions on deaths and confirmed cases for the Covid-19 based on the John Hopkins data. For example, as of today, the USA is on track for 1.6 million deaths by Christmas 2020 and globally we are looking at 4.5 million. These numbers have become better over the last 2 weeks as I have been running them.


## Deploy<a name="deploy"></a>

Update the John Hopkins, Covid-19 data https://github.com/CSSEGISandData/COVID-19 by refreshing your local copy of the GitHub repository. Please point the notebook titled arima_driver.ipynb at your local repository of that data. Simply edit the arg_dict for the jurisdictions and dependent_variable that you want to predict and the notebook will run by calling the following .py files in order.
- load_data: 1) looks at the John Hopkins data and produces a DF suitable for the ARIMA model
- stationarity: 1) shows the raw data, 2) Runs ACF and PACF plots for suggesting initial values of ARIMA p,d,q values
- arima_grid_search: 1) does a grid search and chooses the optimal pdq hyperparameters for the dataset based on the lowest RMSE.
- summarize: 1) produces histograms and density plots that show the bias in the predictions. 2) reruns the predictions bias adjusted.
- test_prediction_save_forecast_1day: 1) creates a plot that shows you the actual (test) vs the predicted (prediction) scores. 2) provides a 1 day in the future forecast (e.g. tomorrow).
- multi_step_forecast: 1) allows you to pick a date any time in the future and predict the dependent_variable (e.g. deaths) up to that date.

All of the code and data required to run this notebook is in included in the GitHub repository.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The John Hopkins data is available under the Creative Commons License. For the particulars of that license please go to https://data.humdata.org/event/covid-19 and search for Creative Commons.

The ARIMA model code was largely taken from this book which I bought "Time Series Forecasting With Python" By Jason Brownlee of Machine Learning Mastery. https://machinelearningmastery.com/make-sample-forecasts-arima-python/ The code was updated to reflect the unique requirements of this Covid-19 analysis. Jason provides code with this book that he explicitly tells you to use and make your own.

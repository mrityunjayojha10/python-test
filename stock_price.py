import datetime
import nsepy
import pandas as pd
import numpy as np
import pytest
import statsmodels
from sklearn import model_selection
from sklearn.linear_model import Ridge
import sys
import scipy as sp
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pickle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

name = sys.argv[1]
stock = nsepy.get_history(symbol=name,
                    start=datetime.date(2015,1,1), 
                    end=datetime.date(2016,1,10))
df = stock[['Close']]
forecast_out = int(1) 
df['Prediction'] = df[['Close']].shift(-forecast_out)
X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:] 
X = X[:-forecast_out]
y = np.array(df['Prediction'])
y = y[:-forecast_out]

def calc_train_error(X_train, y_train, model):
    '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return rmse
    
def calc_validation_error(X_test, y_test, model):
    '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse
    
def calc_metrics(X_train, y_train, X_test, y_test, model):
    '''fits model and returns the RMSE for in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error

X_intermediate, X_test, y_intermediate, y_test = model_selection.train_test_split(X, y, shuffle=True,test_size=0.2, random_state=15)
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X_intermediate, y_intermediate,shuffle=False,test_size=0.25,random_state=2018)
print("hello")
print(X_train.shape)

alphas = [0.001, 0.01, 0.1, 1, 10]
print('All errors are RMSE')
print('-'*76)
error = 10000
for alpha in alphas:
    # instantiate and fit model
    ridge = Ridge(alpha=alpha, fit_intercept=True, random_state=99)
    ridge.fit(X_train, y_train)
    # calculate errors
    new_train_error = np.sqrt(mean_squared_error(y_train, ridge.predict(X_train)))
    new_validation_error = np.sqrt(mean_squared_error(y_validation, ridge.predict(X_validation)))
    new_test_error = np.sqrt(mean_squared_error(y_test, ridge.predict(X_test)))
    if(new_test_error < error):
        error = new_test_error
        model = ridge
    # print errors as report
    print('alpha: {:7} | train error: {:5} | val error: {:6} | test error: {}'.
          format(alpha,
                 round(new_train_error,3),
                 round(new_validation_error,3),
                 round(new_test_error,3)))







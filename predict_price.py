import nsepy
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import sys
from sklearn import preprocessing
import pickle
import datetime

stock_name = sys.argv[1]

stock = nsepy.get_history(symbol=stock_name,
                    start=datetime.date(2015,1,1), 
                    end=datetime.date(2016,1,10))

df = stock[['Close']]
forecast_out = int(1) 
df['Prediction'] = df[['Close']].shift(-forecast_out)
X_test = np.array(df.drop(['Prediction'], 1))
X_test = preprocessing.scale(X_test)

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

y_pred = loaded_model.predict(X_test)

print(y_pred[-1])

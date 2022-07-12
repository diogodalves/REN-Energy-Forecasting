import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

import seaborn as sns
from pylab import rcParams
from matplotlib import rc

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, concatenate, LSTM, Input, Bidirectional, Input, Conv1D, MaxPooling1D, Add, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import callbacks
from keras.utils.vis_utils import plot_model

import pmdarima
from pmdarima.metrics import smape

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_excel('2021_Interface+RESLocal_1.xls', index_col=[0], parse_dates=[0])

df = df[~df.index.duplicated(keep='first')]

def create_features(df):
    df['Date'] = df.index
    df.reset_index()
    df['Date_Parsed'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['DayOfMonth'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.weekofyear
    df['DayOfYearFloat'] = df['DayOfYear'] + df['Hour'] / 24
#     df.drop('Date', axis=1, inplace=True)
    return df

df = create_features(df)

new_df = df.copy()
new_df = new_df.asfreq('H')
new_df = create_features(new_df)

new_df['Interface'] = new_df['Interface'].interpolate(method='polynomial', order=2)
new_df['Local Generation'] = new_df['Local Generation'].interpolate(method='polynomial', order=2)

sarima_model_fitted = pmdarima.auto_arima(new_df[['Interface']],
                                        exogenous=new_df[['Local Generation']],
                                        seasonal=True,
                                        start_p=0, start_q=0, d=1, max_d=3, max_q=5, max_p=5, start_P=0, 
                                        D=1, max_D=3, start_Q=0, max_P=5, max_Q=5, m=24)

p, d, q = sarima_model_fitted.order
sarima_residuals = sarima_model_fitted.arima_res_.resid

print(sarima_model_fitted)

print(sarima_model_fitted.summary())
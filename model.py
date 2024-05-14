import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, GRU, LSTM, Dropout, Dense

# Prepare data
def prepare_data(df):
    training_set = df['2012-01-01':'2016-12-31'].iloc[:, 1:2].values
    test_set = df['2017':].iloc[:, 1:2].values

    sc = MinMaxScaler()
    training_set_scaled = sc.fit_transform(training_set)

    X_train, y_train = [], []
    for i in range(60, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, test_set, sc

# Create model
def create_model(model_type, neurons, input_shape):
    model = Sequential()
    
    if model_type == "LSTM":
        for i, units in enumerate(neurons):
            if i == 0:
                model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
            elif i == len(neurons) - 1:
                model.add(LSTM(units=units))
            else:
                model.add(LSTM(units=units, return_sequences=True))
            model.add(Dropout(0.2))
    elif model_type == "GRU":
        for i, units in enumerate(neurons):
            if i == 0:
                model.add(GRU(units=units, return_sequences=True, input_shape=input_shape))
            elif i == len(neurons) - 1:
                model.add(GRU(units=units))
            else:
                model.add(GRU(units=units, return_sequences=True))
            model.add(Dropout(0.2))
    elif model_type == "SimpleRNN":
        for i, units in enumerate(neurons):
            if i == 0:
                model.add(SimpleRNN(units=units, return_sequences=True, input_shape=input_shape))
            elif i == len(neurons) - 1:
                model.add(SimpleRNN(units=units))
            else:
                model.add(SimpleRNN(units=units, return_sequences=True))
            model.add(Dropout(0.2))
    
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

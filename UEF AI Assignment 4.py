import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt

#LOADING THE DATASET
df = pd.read_csv(
    "../input/lstm-dataset/household_power_consumption.txt",
    sep =";")
#DROPPING NON NUMERIC DATA
df.replace("?", np.nan, inplace =True)
df.dropna(inplace =True)
df.drop(["Date"], 1, inplace = True )

#SAVING MY GLOBAL_ACTIVE_POWER COLUMN AT t
global_active_power_t = np.array(df["Global_active_power"])

#CAUSING A T-1 SHIFT
df = df.shift(-1)
df["global_active_power_t"] = global_active_power_t
df.set_index('Time', inplace =True)

#SCALING THE DATA 
trans = MinMaxScaler()
df = trans.fit_transform(df)
X = df[:,:-1]
y = np.array(df[:,-1:])


# the data available is for 4 years showing every minute so we have to account for that
n_train_mins = (365 * 24 * 60 * 3) + (120 * 24 * 60)

#SPLITTING THE DATA INTO TRAINING AND TEST DATASETS
X_train = X[:n_train_mins,:]
X_test = X[n_train_mins:,:]
y_train = y[:n_train_mins]
y_test = y[n_train_mins:]


# LSTMs take 3d input so we have to transform our data
X_train = X_train.reshape(n_train_mins,1,7)
X_test = X_test.reshape((len(X)-n_train_mins),1,7)

#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# CREATING THE MODEL
model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
Dropout(rate=0.35)
#Dropout(rate = 0.25)
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# FITTING THE MODEL
model.fit(X_train, y_train, epochs=20, batch_size=100,  verbose=2)
#model.fit(X_train,y_train, epochs=20, batch_size=70, validation_data=(X_test, y_test), verbose=2, shuffle=False)
#model.fit(X_train,y_train, epochs=100, batch_size=90, validation_data=(X_test, y_test), verbose=2, shuffle=False)

#PREDICTION MAKING
y_pred = model.predict(X_test, verbose=2)
print(y_pred)

# PLOTTING THE FIRST TEN PREDICTIONS AGAINST THE ACTUAL VALUE
plt.plot(y_pred[:10], label='Prediction')
plt.plot(y_test[:10], label='Actual Value')
plt.legend()
plt.show()

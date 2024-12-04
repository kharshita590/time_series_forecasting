import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout
from tensorflow.keras.regularizers import L1L2
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('DailyDelhiClimateTest.csv', parse_dates=['date'], dayfirst=True)
temp_values = data['meantemp'].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_temp = scaler.fit_transform(temp_values)
train_size = int(len(scaled_temp) * 0.6)
test_size = len(scaled_temp)-train_size
train,test = scaled_temp[0:train_size,:] , scaled_temp[train_size:len(scaled_temp),:]
def create_dataset(dataset,look_back=1):
  X,Y=[],[]
  for i in range(len(dataset)-look_back-1):
    a=dataset[i:(i+look_back),0]
    X.append(a)
    Y.append(dataset[i+look_back,0])
  return np.array(X), np.array(Y)
look_back=7

X_train,Y_train = create_dataset(train,look_back)
X_test,Y_test=create_dataset(test,look_back)
X_train = X_train.reshape(X_train.shape[0],look_back,-1)
X_test = X_test.reshape(X_test.shape[0],look_back,-1)
model = Sequential()
model.add(LSTM(50,input_shape=(look_back,1),return_sequences=True,kernel_regularizer=L1L2(l1=1e-5,l2=1e-4)))
model.add(Dropout(0.3))
model.add(LSTM(25,input_shape=(look_back,1),return_sequences=False,kernel_regularizer=L1L2(l1=1e-5,l2=1e-4)))
model.add(Dropout(0.4))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
early_stopping = EarlyStopping(monitor='val_loss', patience=7)
model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=2, callbacks=[early_stopping], validation_split=0.4)
train_predict = scaler.inverse_transform(model.predict(X_train))
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(model.predict(X_test))
Y_test = scaler.inverse_transform([Y_test])
train_rmse = np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))
train_mae = mean_absolute_error(Y_train[0], train_predict[:,0])
test_rmse = np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))
test_mae = mean_absolute_error(Y_test[0], test_predict[:,0])

import numpy as np
from sklearn.preprocessing import MinMaxScaler
def predict_temperature(input_sequence, model, look_back=7, scaler=None):
    if len(input_sequence) != look_back:
        raise ValueError(f"Input sequence must be of length {look_back}.")
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_sequence = np.array(input_sequence).reshape(-1, 1)
    input_sequence_scaled = scaler.fit_transform(input_sequence)
    input_sequence_scaled = input_sequence_scaled.reshape(1, look_back, 1)
    predicted_scaled = model.predict(input_sequence_scaled)
    predicted_temp = scaler.inverse_transform(predicted_scaled)

    return predicted_temp[0][0]
print(predict_temperature([23,45,65,26,45,32,34],model))

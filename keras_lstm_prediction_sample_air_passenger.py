# coding:utf-8
# http://sweng.web.fc2.com/ja/program/python/time-series-forecast-lstm.html

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd
from matplotlib import pylab as plt
#from matplotlib import pyplot
import seaborn as sns
import pickle
#%matplotlib inline
sns.set()

unit_num = 100
train_samples = 72
input_data_len = 24
output_data_len = 12
epochs = 50

df = pd.read_csv('AirPassengers.csv', index_col='Month', dtype={1: 'float'})
ts = df['#Passengers']

x = []  # train
y = []  # test (answer)
# xは24要素のリストを、1要素ずつスライドさせながら72個とっている
# yは同様にして、時系列的にx内の24要素のリストに続く12要素のリストを72個とっている
# 結果的にxとyの同一インデックスのリストは <24の時系列データ(入力データ)> - <入力データに続く12の時系列データ（教師データ）> となっている
for i in range(0, train_samples):
    tmpX = []
    for j in range(0, input_data_len):
        tmpX.append(ts[i + j])
    x.append(tmpX)

    tmpY = []
    for j in range(0, output_data_len):
        tmpY.append(ts[input_data_len + i + j])
    y.append(tmpY)

x = np.array(x)
y = np.array(y)
x = x.reshape((x.shape[0], x.shape[1], 1))
y = y.reshape((y.shape[0], y.shape[1], 1))

m = Sequential()
# 入力データ数が input_data_len なので、input_shapeの値は(input_data_len,1)
m.add(LSTM(unit_num, activation='relu', input_shape=(input_data_len, 1)))
# 予測範囲は output_data_lenステップなので、RepeatVectoorにoutput_data_lenを指定
m.add(RepeatVector(output_data_len))
m.add(LSTM(unit_num, activation='relu', return_sequences=True))
m.add(TimeDistributed(Dense(1)))
m.compile(optimizer='adam', loss='mse')
#m.fit(x, y, epochs=1000, verbose=1)
m.fit(x, y, epochs=epochs, verbose=1)

# データ60番～83番から、次の一年(84番～95番)を予測
input_start_idx = 60
input = np.array(ts[input_start_idx:input_start_idx + input_data_len])
input = input.reshape((1, input_data_len, 1))
yhat = m.predict(input)

# 可視化用に、予測結果yhatを、配列predictに格納
predict = []
for i in range(0, output_data_len):
    predict.append(yhat[0][i])

# 比較するために実データをプロット
plt.plot(ts)

# 予測したデータをプロット
predicted_start_idx = input_start_idx + input_data_len
xdata = np.arange(predicted_start_idx, predicted_start_idx + output_data_len, 1)
plt.plot(xdata, predict, 'r')
plt.show()


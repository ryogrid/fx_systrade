# coding:utf-8
# http://sweng.web.fc2.com/ja/program/python/time-series-forecast-lstm.html

import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd
from matplotlib import pylab as plt
#from matplotlib import pyplot
import seaborn as sns
import pickle
import os
#%matplotlib inline
sns.set()

unit_num = 100
train_samples = 2100 #  #72
input_data_len = 48 #24
output_data_len = 12 #12
future_period = 15 #20 #入力する時系列データから何要素離れたデータを予測するか
epochs = 50 #100 #50

# 学習結果をテストする際のパラメータ
# test_period = 100 # 予測する要素数

# df = pd.read_csv('AirPassengers.csv', index_col='Month', dtype={1: 'float'})
# ts = df['#Passengers']

exchange_rates = None
with open("./exchange_rates.pickle", 'rb') as f:
    exchange_rates = pickle.load(f)
# 先頭3000要素のみ使う
exchange_rates = exchange_rates[:3000]

x = []  # train
y = []  # test (answer)
if os.path.exists("./x.pickle"):
    with open("./x.pickle", 'rb') as f:
        x = pickle.load(f)
    with open("./y.pickle", 'rb') as f:
        y = pickle.load(f)
else:
    # xは24要素のリストを、1要素ずつスライドさせながら72個とっている
    # yは同様にして、時系列的にx内の24要素のリストに続く12要素のリストを72個とっている
    # 結果的にxとyの同一インデックスのリストは <24の時系列データ(入力データ)> - <入力データに続く12の時系列データ（教師データ）> となっている
    for i in range(0, train_samples):
        tmpX = []
        for j in range(0, input_data_len):
            tmpX.append(exchange_rates[i + j])
        x.append(tmpX)

        tmpY = []
        for j in range(0, output_data_len):
            tmpY.append(exchange_rates[input_data_len + i + j + future_period])
        y.append(tmpY)
    with open("./x.pickle", 'wb') as f:
        pickle.dump(x, f)
    with open("./y.pickle", 'wb') as f:
        pickle.dump(y, f)

print("train data preparation finished.")

x = np.array(x)
y = np.array(y)
x = x.reshape((x.shape[0], x.shape[1], 1))
y = y.reshape((y.shape[0], y.shape[1], 1))

m = None
if os.path.exists("./m_nw.json"):
    with open("./m_nw.json", "r") as f:
        m = model_from_json(f.read())
    m.compile(optimizer='adam', loss='mse')
    m.load_weights("./m_weights.hd5")
else:
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

    with open("./m_nw.json", "w") as f:
        f.write(m.to_json())
    m.save_weights("./m_weights.hd5")

print("learninged model preparation finshed.")

# input_start_idx:input_start_idx + input_data_lenのデータから
# そこからfuture_period足離れたところ output_data_len個の要素を予測する
input_start_idx = 2800
input = np.array(exchange_rates[input_start_idx:input_start_idx + input_data_len])
input = input.reshape((1, input_data_len, 1))
yhat = m.predict(input)
# 可視化用に、予測結果yhatを、配列predictに格納
predict = []
for i in range(0, output_data_len):
    predict.append(yhat[0][i])

# predict = []
# test_start_indx = 2800
# # データ test_start_index + input_data_lenから、1要素ずつスライドさせて得られるtest_period個のリストから、
# # データ test_start_index + input_data_len + future_period から test_period個のデータを予測する
# for idx in range(test_start_indx, test_start_indx + test_period):
#     input_start_idx = idx
#     input = np.array(exchange_rates[input_start_idx:input_start_idx + input_data_len])
#     input = input.reshape((1, input_data_len, 1))
#     yhat = m.predict(input)
#     # 可視化用に、予測結果yhatを、配列predictに格納
#     predict.append(yhat[0][0])

# 比較するために実データをプロット
plt.plot(exchange_rates)

# 予測したデータをプロット
predicted_start_idx = input_start_idx + input_data_len + future_period
xdata = np.arange(predicted_start_idx, predicted_start_idx + output_data_len, 1)
plt.plot(xdata, predict, 'r')
plt.show()

# # 予測したデータをプロット
# # 以下はoutput_data_lenが1である前提での計算式
# predicted_start_idx = test_start_indx + input_data_len + future_period
# xdata = np.arange(predicted_start_idx, predicted_start_idx + test_period, 1)
# plt.plot(xdata, predict, 'r')
# plt.show()


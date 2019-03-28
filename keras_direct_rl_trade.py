#!/usr/bin/python
from __future__ import absolute_import

import numpy as np
import scipy.sparse
import xgboost as xgb
import pickle
import talib as ta

import pandas as pd

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

INPUT_LEN = 12 # 1h
OUTPUT_LEN = 5
TRAINDATA_DIV = 10
HALF_SPREAD = 0.0015

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler
    
def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder


"""
main
"""
rates_fd = open('./hoge.csv', 'r')
exchange_dates = []
exchange_rates = []
exchange_rates_diff = []
prev = 100
for line in rates_fd:
    splited = line.split(",")
    if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>"and splited[0] != "204/04/26" and splited[0] != "20004/04/26":
        time = splited[0].replace("/", "-") + " " + splited[1]
        val = float(splited[2])
        exchange_dates.append(time)
        exchange_rates.append(val)
        exchange_rates_diff.append(val - prev)
        prev = val

data_len = len(exchange_rates)
train_len = len(exchange_rates)/TRAINDATA_DIV

print "data size: " + str(data_len)
print "train len: " + str(train_len)

if False:
    dump_fd = open("./keras_direct2.model", "r")
    model = model_from_json(dump_fd.read())
    model.load_weights("./keras_direct2.weight")
    
if True: ### training start
    tr_input_mat = []
    tr_angle_mat = []
    for ii in xrange(1000, train_len):
        tmp_arr = []
        cur_pos = 0
        for jj in xrange(INPUT_LEN):
            tmp_arr.append(exchange_rates_diff[ii-INPUT_LEN+jj])
        
        tr_input_mat.append(tmp_arr)

        long_case = (exchange_rates[ii+OUTPUT_LEN] - HALF_SPREAD) - (exchange_rates[ii] + HALF_SPREAD)
        short_case =  (exchange_rates[ii+OUTPUT_LEN] + HALF_SPREAD) - (exchange_rates[ii] - HALF_SPREAD)
        if long_case >= 0.1:
            tr_angle_mat.append(0)
        elif short_case <= -0.1:
            tr_angle_mat.append(1)
        else:
            tr_angle_mat.append(2)
        # label 0 means can't win
        
    X = np.array(tr_input_mat, dtype=np.float32)
    Y = np.array(tr_angle_mat, dtype=np.float32)

    X, scaler = preprocess_data(X)
    Y, encoder = preprocess_labels(Y)

    np.random.seed(1337) # for reproducibility

    nb_classes = 3
    print(nb_classes, 'classes')

    dims = X.shape[1]
    print(dims, 'dims')

    neuro_num = dims
    
    # setup deep NN
    model = Sequential()
    model.add(Dense(neuro_num,input_shape=(dims,), init='uniform', activation="relu"))
    # model.add(BatchNormalization((neuro_num,)))
    # model.add(Dropout(0.5))

    # model.add(Dense(neuro_num/2, init='uniform', activation="relu"))
    # model.add(BatchNormalization((neuro_num/2,)))
    # model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes, init='uniform', activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy', optimizer="adam")
    
    print("Training model...")
    model.fit(X, Y, nb_epoch=10000, batch_size=100, validation_split=0.15)

    dump_fd = open("./keras_direct2.model", "w")
    model_json_str = model.to_json()
    dump_fd.write(model_json_str)
    model.save_weights("keras_direct2.weight")
    
### training end

# trade
portfolio = 1000000
LONG = 1
SHORT = 2
NOT_HAVE = 3
pos_kind = NOT_HAVE


positions = 0
trade_val = -1

pos_cont_count = 0
for window_s in xrange((data_len - train_len) - (OUTPUT_LEN)):
    current_spot = train_len + window_s + OUTPUT_LEN
        
    if pos_kind != NOT_HAVE:
        if pos_cont_count >= (OUTPUT_LEN-1):
            if pos_kind == LONG:
                pos_kind = NOT_HAVE
                portfolio = positions * (exchange_rates[current_spot] - HALF_SPREAD)
                print exchange_dates[current_spot] + " " + str(portfolio)
            elif pos_kind == SHORT:
                pos_kind = NOT_HAVE
                portfolio += positions * trade_val - positions * (exchange_rates[current_spot] + HALF_SPREAD)
                print exchange_dates[current_spot] + " " + str(portfolio)
            pos_cont_count = 0
        else:
            pos_cont_count += 1
            continue

    # prediction    
    ts_input_mat = []
    tmp_arr = []
    for jj in xrange(INPUT_LEN):
            tmp_arr.append(exchange_rates_diff[current_spot-INPUT_LEN+jj])
    ts_input_mat.append(tmp_arr)    

    ts_input_arr = np.array(ts_input_mat)

    X_test = np.array(ts_input_arr, dtype=np.float32)
    X_test, _ = preprocess_data(X_test, scaler)
    # X_test, _ = preprocess_data(X_test)
    
    proba = model.predict_proba(X_test, verbose=0)

    if pos_kind == NOT_HAVE:
        if proba[0][0] >= 0.8 :
           pos_kind = LONG
           positions = portfolio / (exchange_rates[current_spot] + HALF_SPREAD)
           trade_val = exchange_rates[current_spot] + HALF_SPREAD
        elif proba[0][1] >= 0.8:
           pos_kind = SHORT
           positions = portfolio / (exchange_rates[current_spot] - HALF_SPREAD)
           trade_val = exchange_rates[current_spot] - HALF_SPREAD

#!/usr/bin/python
import numpy as np
import scipy.sparse
import xgboost as xgb
import pickle
import talib as ta
from datetime import datetime as dt
import pytz

INPUT_LEN = 1
OUTPUT_LEN = 5
TRAINDATA_DIV = 10
HALF_SPREAD = 0.0015

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
    if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>"and splited[0] != "204/04/26" and splited[0] != "20004/04/26": # and (not is_weekend(splited[0])):
        time = splited[0].replace("/", "-") + " " + splited[1]
        val = float(splited[2]) #for hoge.csv
        exchange_dates.append(time)
        exchange_rates.append(val)
        exchange_rates_diff.append(val - prev)
        prev = val

data_len = len(exchange_rates)
# train_len = len(exchange_rates)/TRAINDATA_DIV
train_len = 1000

print "data size: " + str(data_len)
print "train len: " + str(train_len)

if True:
    bst = xgb.Booster({'nthread':4})
    bst.load_model("./xdirect2.model") 

if False: ### training start
    tr_input_mat = []
    tr_angle_mat = []
    prev_pred = 0
    for ii in xrange(INPUT_LEN, train_len):
        print("step " + str(ii))
        tmp_arr = [prev_pred]
        cur_pos = 0
        for jj in xrange(INPUT_LEN):
            tmp_arr.append(exchange_rates_diff[ii-INPUT_LEN+jj])
        tr_input_mat.append(tmp_arr)

        long_case = (exchange_rates[ii+OUTPUT_LEN] - HALF_SPREAD) - (exchange_rates[ii] + HALF_SPREAD)
        if long_case >= 0.1:
            tr_angle_mat.append(1)
        else:
            tr_angle_mat.append(0)
        
        tr_input_arr = np.array(tr_input_mat)
        tr_angle_arr = np.array(tr_angle_mat)
        dtrain = xgb.DMatrix(tr_input_arr, label=tr_angle_arr)
        param = {'max_depth':6, 'eta':0.2, 'subsumble':0.5, 'silent':1, 'objective':'binary:logistic' }
        watchlist  = [(dtrain,'train')]
        num_round =  3000 #3000 #10 #3000 # 1000
        # bst = xgb.train(param, dtrain, num_round, watchlist)
        bst = xgb.train(param, dtrain, num_round)        

        # prediction    
        ts_input_mat = []
        ts_input_mat.append(tmp_arr)    

        ts_input_arr = np.array(ts_input_mat)
        dtest = xgb.DMatrix(ts_input_arr)
        pred = bst.predict(dtest)
        predicted_prob = pred[0]

        if predicted_prob >= 0.90:
            prev_pred = 1
        else:
            prev_pred = 0

    
bst.save_model('./xdirect2.model')
    
### training end

# trade
portfolio = 1000000
LONG = 1
SHORT = 2
NOT_HAVE = 3
pos_kind = NOT_HAVE
SONKIRI_RATE = 0.05
RIKAKU_PIPS = 0.60

positions = 0

trade_val = -1

pos_cont_count = 0
won_pips = 0
prev_pred = 0

for window_s in xrange((data_len - train_len) - (OUTPUT_LEN)):
    # print("trading")    
    current_spot = train_len + window_s + OUTPUT_LEN

    if pos_kind != NOT_HAVE:
        if pos_cont_count >= (OUTPUT_LEN-1):
            if pos_kind == LONG:
                pos_kind = NOT_HAVE
                portfolio = positions * (exchange_rates[current_spot] - HALF_SPREAD)
                diff = (exchange_rates[current_spot] - HALF_SPREAD) - trade_val
                won_pips += diff
                print(str(diff) + "pips " + str(won_pips) + "pips")                
                print exchange_dates[current_spot] + " " + str(portfolio)
            pos_cont_count = 0
        else:
            pos_cont_count += 1
            continue
    
    # prediction    
    ts_input_mat = []
    tmp_arr = [prev_pred]
    for jj in xrange(INPUT_LEN):
            tmp_arr.append(exchange_rates_diff[current_spot-INPUT_LEN+jj])
    ts_input_mat.append(tmp_arr)    

    ts_input_arr = np.array(ts_input_mat)
    dtest = xgb.DMatrix(ts_input_arr)

    pred = bst.predict(dtest)

    predicted_prob = pred[0]

    if pos_kind == NOT_HAVE:
        if predicted_prob >= 0.90:
            pos_kind = LONG
            positions = portfolio / (exchange_rates[current_spot] + HALF_SPREAD)
            trade_val = exchange_rates[current_spot] + HALF_SPREAD
            prev_pred = 1
        else:
            prev_pred = 0

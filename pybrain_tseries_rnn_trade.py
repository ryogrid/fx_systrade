#!/usr/bin/python
from __future__ import absolute_import

import numpy as np
import scipy.sparse
import xgboost as xgb
import pickle
import pandas as pd
import sys
import json
import random
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.structure import RecurrentNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer    
from datetime import datetime as dt
import pytz

INPUT_LEN = 25
OUTPUT_LEN = 5
TRAINDATA_DIV = 10
CHART_TYPE_JDG_LEN = 25

def construct_network(input_len, hidden_nodes, is_elman=True):
    n = RecurrentNetwork()
    n.addInputModule(LinearLayer(input_len, name="i"))
    n.addModule(BiasUnit("b"))
    n.addModule(SigmoidLayer(hidden_nodes, name="h"))
    n.addOutputModule(LinearLayer(1, name="o"))

    n.addConnection(FullConnection(n["i"], n["h"]))
    n.addConnection(FullConnection(n["b"], n["h"]))
    n.addConnection(FullConnection(n["b"], n["o"]))
    n.addConnection(FullConnection(n["h"], n["o"]))

    if is_elman:
        # Elman (hidden->hidden)
        n.addRecurrentConnection(FullConnection(n["h"], n["h"]))
    else:
        # Jordan (out->hidden)
        n.addRecurrentConnection(FullConnection(n["o"], n["h"]))

    n.sortModules()
    n.reset()

    return n

def get_vorarity(price_arr, cur_pos, period = None):
    tmp_arr = []
    prev = -1
    for val in price_arr[cur_pos-CHART_TYPE_JDG_LEN:cur_pos]:
        if prev == -1:
            tmp_arr.append(0)
        else:
            tmp_arr.append(val - prev)
        prev = val
        
    return np.std(tmp_arr)

# 0->flat 1->upper line 2-> downer line 3->above is top 4->below is top
def judge_chart_type(data_arr):
    max_val = 0
    min_val = float("inf")

    last_idx = len(data_arr)-1
    
    for idx in xrange(len(data_arr)):
        if data_arr[idx] > max_val:
            max_val = data_arr[idx]
            max_idx = idx

        if data_arr[idx] < min_val:
            min_val = data_arr[idx]
            min_idx = idx


    if max_val == min_val:
        return 0
    
    if min_idx == 0 and max_idx == last_idx:
        return 1

    if max_idx == 0 and min_idx == last_idx:
        return 2

    if max_idx != 0 and max_idx != last_idx and min_idx != 0 and min_idx != last_idx:
        return 0
    
    if max_idx != 0 and max_idx != last_idx:
        return 3

    if min_idx != 0 and min_idx != last_idx:
        return 4
        
    return 0

def diff_to_code(diff):
    if diff > 1:
        return 1
    elif diff > 0.5:
        return 2
    elif diff > 0.3:
        return 3
    elif diff > 0.2:
        return 4
    elif diff > 0.1:
        return 5
    elif diff > 0.05:
        return 6
    elif diff > 0.03:
        return 7
    elif diff > 0.02:
        return 8
    elif diff > 0.015:
        return 9
    elif diff > 0.01:
        return 10
    elif diff > 0.005:
        return 11
    elif diff > 0:
        return 12
    elif diff > -0.005:
        return 13
    elif diff > -0.01:
        return 14
    elif diff > -0.015:
        return 15
    elif diff > -0.02:
        return 16
    elif diff > -0.03:
        return 17
    elif diff > -0.05:
        return 18
    elif diff > -0.1:
        return 19
    elif diff > -0.2:
        return 20
    elif diff > -0.3:
        return 21
    elif diff > -0.5:
        return 22
    elif diff > -1:
        return 23
    else:
        return 24

def make_code_arr(rates_arr, start_idx, length):
    ret_arr = []

    for idx in xrange(start_idx, start_idx + length):
        diff = rates_arr[idx+1] - rates_arr[idx]
        ret_arr.append(diff_to_code(diff))

    print(str(ret_arr))
    return ret_arr

def is_weekend(date_str):
    tz = pytz.timezone('Asia/Tokyo')
    tdatetime = dt.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    tz_time = tz.localize(tdatetime)
    london_tz = pytz.timezone('Europe/London')
    london_time = tz_time.astimezone(london_tz)
    week = london_time.weekday()
    return (week == 5 or week == 6)

"""
main
"""
hidden_nodes = 500
parameters = {}

rates_fd = open('./hoge.csv', 'r')
exchange_dates = []
exchange_rates = []
for line in rates_fd:
    splited = line.split(",")
    if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>"and splited[0] != "204/04/26" and splited[0] != "20004/04/26" and (not is_weekend(splited[0])):
        time = splited[0].replace("/", "-") + " " + splited[1]
        val = float(splited[2])
        exchange_dates.append(time)
        exchange_rates.append(val)

reverse_exchange_rates = []
prev_org = -1
prev = -1
for rate in exchange_rates:
    if prev_org != -1:
        diff = rate - prev_org
        reverse_exchange_rates.append(prev - diff)
        prev_org = rate
        prev = prev - diff
    else:
        reverse_exchange_rates.append(rate)
        prev_org = rate
        prev = rate
    
data_len = len(exchange_rates)
train_len = len(exchange_rates)/TRAINDATA_DIV

print "data size: " + str(data_len)
print "train len: " + str(train_len)

if False:
    dump_fd = open("./xg_tseries.dump", "r")
    bst = pickle.load(dump_fd)

if True: ### training start
    rnn_net = construct_network(INPUT_LEN, hidden_nodes, True)
    tr_input_mat = []
    tr_angle_mat = []
    t_ds_list = []    
    for i in xrange(1000, train_len, OUTPUT_LEN):
        input_list1 = make_code_arr(exchange_rates, i, INPUT_LEN)
        input_list2 = make_code_arr(reverse_exchange_rates, i, INPUT_LEN)

        output_list1 = []
        output_list2 = []        
        tmp = (exchange_rates[i+OUTPUT_LEN] - exchange_rates[i])/float(OUTPUT_LEN)
        if tmp >= 0:
            output_list1.append(1)
        else:
            output_list1.append(0)
            
        tmp = (reverse_exchange_rates[i+OUTPUT_LEN] - reverse_exchange_rates[i])/float(OUTPUT_LEN)
        if tmp >= 0:
            output_list2.append(1)
        else:
            output_list2.append(0)

        for i in xrange(0, 10):
            t_ds_list.append((input_list1, output_list1))
            t_ds_list.append((input_list2, output_list2))            

    t_ds = SupervisedDataSet(INPUT_LEN, 1)
    random.shuffle(t_ds_list)    
    for data in t_ds_list:
        t_ds.addSample(data[0], data[1])

    trainer = BackpropTrainer(rnn_net, **parameters)
    trainer.setData(t_ds)
    trainer.train()    
    
### training end

# trade
portfolio = 1000000
LONG = 1
SHORT = 2
NOT_HAVE = 3
pos_kind = NOT_HAVE
HALF_SPREAD = 0 #0.0015
SONKIRI_RATE = 0.05

positions = 0

trade_val = -1

pos_cont_count = 0
for window_s in xrange((data_len - train_len) - (OUTPUT_LEN)):
    current_spot = train_len + window_s + OUTPUT_LEN
    skip_flag = False

    #sonkiri
    if pos_kind != NOT_HAVE:
        if pos_kind == LONG:
            cur_portfo = positions * (exchange_rates[current_spot] - HALF_SPREAD)
        elif pos_kind == SHORT:
            cur_portfo = portfolio + (positions * trade_val - positions * (exchange_rates[current_spot] + HALF_SPREAD))
        if (cur_portfo - portfolio)/portfolio < -1*SONKIRI_RATE:
            portfolio = cur_portfo
            pos_kind = NOT_HAVE
            continue
        
    # # chart_type = 0
    # chart_type = judge_chart_type(exchange_rates[current_spot-CHART_TYPE_JDG_LEN:current_spot])
    # if chart_type != 1 and chart_type != 2:
    #     skip_flag = True
    #     if pos_kind != NOT_HAVE:
    #         # if liner trend keep position
    #         continue
        
    # print "state1 " + str(pos_kind)    
    if pos_kind != NOT_HAVE:
        # print "pos_cont_count " + str(pos_cont_count)
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

    # try trade in only linear chart case
    vorarity = get_vorarity(exchange_rates, current_spot)
    if vorarity >= 0.07:
        skip_flag = True
    
    # prediction    
    rnn_net.reset()
    ts_input_mat = make_code_arr(exchange_rates, i, INPUT_LEN)

    predicted_prob = rnn_net.activate(ts_input_mat)
    print(str(predicted_prob))
    
    # Print "state2 " + str(pos_kind)
    # print "predicted_prob " + str(predicted_prob)
    # print "skip_flag:" + str(skip_flag)
    if pos_kind == NOT_HAVE and skip_flag == False:
        if predicted_prob >= 0.9 :
           pos_kind = LONG
           positions = portfolio / (exchange_rates[current_spot] + HALF_SPREAD)
           trade_val = exchange_rates[current_spot] + HALF_SPREAD
        elif predicted_prob <= 0.1:
           pos_kind = SHORT
           positions = portfolio / (exchange_rates[current_spot] - HALF_SPREAD)
           trade_val = exchange_rates[current_spot] - HALF_SPREAD

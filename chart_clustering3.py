#!/usr/bin/python
import math
import matplotlib.pylab as plt

# http://alexminnaar.com/time-series-classification-and-clustering-with-python.html

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
        
    

"""
main
"""
SERIES_LEN = 25
DATA_NUM = 250
MA_PERIOD = 5

rates_fd = open('./hoge.csv', 'r')
exchange_rates = []
for line in rates_fd:
    splited = line.split(",")
    if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>"and splited[0] != "204/04/26" and splited[0] != "20004/04/26":
        time = splited[0].replace("/", "-") + " " + splited[1]
        val = float(splited[2])
        exchange_rates.append(val-110)

ma_vals = []        
for idx in xrange(MA_PERIOD, len(exchange_rates)):
    ma_vals.append(sum(exchange_rates[idx:idx+MA_PERIOD])/MA_PERIOD)
    
tmp_arr = []
to_out_arr = []
for window_s in xrange(1, DATA_NUM):
    current_spot = SERIES_LEN * window_s
    print str(judge_chart_type(ma_vals[current_spot:current_spot + SERIES_LEN]))    
    for val in ma_vals[current_spot:current_spot + SERIES_LEN]:
        print str(val) + "," ,
    print("")

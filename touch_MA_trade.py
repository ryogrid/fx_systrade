#!/usr/bin/python
import numpy as np
import scipy.sparse
import xgboost as xgb
import pickle
import talib as ta

def merge_csv(out_fname, input_files):
    frslt = open('./hoge.csv', 'w')        
    frslt.write("Date Time,Open,High,Low,Close,Volume,Adj Close\n")

    for iname in input_files:
        fd = open(iname, 'r')
        for trxline in fd:
            splited = trxline.split(",")
            if splited[0] != "<DTYYYYMMDD>" and splited[0] != "204/04/26" and splited[0] != "20004/04/26":
                time = splited[0].replace("/", "-") + " " + splited[1]
                val = splited[2]

                frslt.write(str(time) + "," + str(val) + "," + \
                            str(val) + "," + str(val) + \
                            "," + str(val) + ",1000000,"+ str(val) + "\n")

    frslt.close()

def get_ma_list(price_arr, cur_pos, period = 20):
    if cur_pos <= (period + 2):
        s = 0
    else:
        s = cur_pos - (period + 2)
    tmp_arr = price_arr[s:cur_pos]
    tmp_arr.reverse()
    prices = np.array(tmp_arr, dtype=float)

    return ta.SMA(prices, timeperiod = period)

def touch_top(rates, ma_list, cur):
    if rates[cur - 2] < ma_list[-3] and rates[cur - 1] > ma_list[-2] and rates[cur] < ma_list[-1]:
        return True
    else:
        return False

def touch_bottom(rates, ma_list, cur):
    if rates[cur - 2] > ma_list[-3] and rates[cur - 1] < ma_list[-2] and rates[cur] > ma_list[-1]:
        return True
    else:
        return False

def cross_above(rates, ma_list, cur):
    if rates[cur - 1] < ma_list[-2] and rates[cur] > ma_list[-1]:
        return True
    else:
        return False
    
def cross_below(rates, ma_list, cur):
    if rates[cur - 1] > ma_list[-2] and rates[cur] < ma_list[-1]:
        return True
    else:
        return False
    
"""
main
"""
rates_fd = open('./hoge.csv', 'r')
exchange_dates = []
exchange_rates = []
for line in rates_fd:
    splited = line.split(",")
    if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>"and splited[0] != "204/04/26" and splited[0] != "20004/04/26":
        time = splited[0].replace("/", "-") + " " + splited[1]
        val = float(splited[2])
        exchange_dates.append(time)
        exchange_rates.append(val)

data_len = len(exchange_rates)

print "data size: " + str(data_len)

portfolio = 1000000
LONG = 1
SHORT = 2
NOT_HAVE = 3
pos_kind = NOT_HAVE
HALF_SPREAD = 0.002

positions = 0

trade_val = -1
for cur in xrange(21, data_len):

    print "state " + str(pos_kind)
    ma_list = get_ma_list(exchange_rates, cur)
#    ma_list_5 = get_ma_list(exchange_rates, cur, 5)
#    print ma_list
    
    if pos_kind == NOT_HAVE:
        if touch_top(exchange_rates, ma_list, cur):
           pos_kind = LONG
           positions = portfolio / (exchange_rates[cur] + HALF_SPREAD)
           trade_val = exchange_rates[cur] + HALF_SPREAD
        elif touch_bottom(exchange_rates, ma_list, cur):
           pos_kind = SHORT
           positions = portfolio / (exchange_rates[cur] - HALF_SPREAD)
           trade_val = exchange_rates[cur] - HALF_SPREAD
    else:
        if pos_kind == LONG and touch_bottom(exchange_rates, ma_list, cur):# or cross_below(exchange_rates, ma_list_5, cur)):
            pos_kind = NOT_HAVE
            portfolio = positions * (exchange_rates[cur] - HALF_SPREAD)
        elif pos_kind == SHORT and touch_top(exchange_rates, ma_list, cur):# or cross_above(exchange_rates, ma_list_5, cur)):
            pos_kind = NOT_HAVE
            portfolio += positions * trade_val - positions * (exchange_rates[cur] + HALF_SPREAD)

    print exchange_dates[cur] + " " + str(portfolio)

# coding:utf-8
import numpy as np
import scipy.sparse
import pickle
import talib as ta
from datetime import datetime as dt
import pytz
import os
import sys

import time

INPUT_LEN = 1
SLIDE_IDX_NUM_AT_GEN_INPUTS_AND_COLLECT_LABELS = 1 #5
PREDICT_FUTURE_LEGS = 5
COMPETITION_DIV = True
COMPETITION_TRAIN_DATA_NUM = 223954 # 3years (test is 5 years)

TRAINDATA_DIV = 2
CHART_TYPE_JDG_LEN = 25

VALIDATION_DATA_RATIO = 1.0 # rates of validation data to (all data - train data)
DATA_HEAD_ASOBI = 200

FEATURE_NAMES = ["current_rate", "diff_ratio_between_previous_rate", "rsi", "ma", "ma_kairi", "bb_1", "bb_2", "ema", "cci", "mo","vorariity", "macd", "chart_type"]

tr_input_arr = None
tr_angle_arr = None
val_input_arr = None
val_angle_arr = None

exchange_dates = None
exchange_rates = None
reverse_exchange_rates = None

# chart_filter_type_long = [2]
# chart_filter_type_short = [1]

# 0->flat 1->upper line 2-> downer line 3->above is top 4->below is top
def judge_chart_type(data_arr):
    max_val = 0
    min_val = float("inf")

    last_idx = len(data_arr)-1

    for idx in range(len(data_arr)):
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

def get_rsi(price_arr, cur_pos, period = 40):
    if cur_pos <= period:
#        s = 0
        return 0
    else:
        s = cur_pos - (period + 1)
    tmp_arr = price_arr[s:cur_pos]
    tmp_arr.reverse()
    prices = np.array(tmp_arr, dtype=float)

    return ta.RSI(prices, timeperiod = period)[-1]

def get_ma(price_arr, cur_pos, period = 20):
    if cur_pos <= period:
        s = 0
    else:
        s = cur_pos - period
    tmp_arr = price_arr[s:cur_pos]
    tmp_arr.reverse()
    prices = np.array(tmp_arr, dtype=float)

    return ta.SMA(prices, timeperiod = period)[-1]

def get_ma_kairi(price_arr, cur_pos, period = None):
    ma = get_ma(price_arr, cur_pos)
    return ((price_arr[cur_pos] - ma) / ma) * 100.0
    return 0

def get_bb_1(price_arr, cur_pos, period = 40):
    if cur_pos <= period:
        s = 0
    else:
        s = cur_pos - period
    tmp_arr = price_arr[s:cur_pos]
    tmp_arr.reverse()
    prices = np.array(tmp_arr, dtype=float)

    return ta.BBANDS(prices, timeperiod = period)[0][-1]

def get_bb_2(price_arr, cur_pos, period = 40):
    if cur_pos <= period:
        s = 0
    else:
        s = cur_pos - period
    tmp_arr = price_arr[s:cur_pos]
    tmp_arr.reverse()
    prices = np.array(tmp_arr, dtype=float)

    return ta.BBANDS(prices, timeperiod = period)[2][-1]

def get_ema(price_arr, cur_pos, period = 20):
    if cur_pos <= period:
        s = 0
    else:
        s = cur_pos - period
    tmp_arr = price_arr[s:cur_pos]
    tmp_arr.reverse()
    prices = np.array(tmp_arr, dtype=float)

    return ta.EMA(prices, timeperiod = period)[-1]


# def get_ema_rsi(price_arr, cur_pos, period = None):
#     return 0

def get_cci(price_arr, cur_pos, period = None):
    return 0

def get_mo(price_arr, cur_pos, period = 20):
    if cur_pos <= (period + 1):
#        s = 0
        return 0
    else:
        s = cur_pos - (period + 1)
    tmp_arr = price_arr[s:cur_pos]
    tmp_arr.reverse()
    prices = np.array(tmp_arr, dtype=float)

    return ta.CMO(prices, timeperiod = period)[-1]

def get_po(price_arr, cur_pos, period = 10):
    if cur_pos <= period:
        s = 0
    else:
        s = cur_pos - period
    tmp_arr = price_arr[s:cur_pos]
    tmp_arr.reverse()
    prices = np.array(tmp_arr, dtype=float)

    return ta.PPO(prices)[-1]

def get_vorarity(price_arr, cur_pos, period = None):
    tmp_arr = []
    prev = -1.0
    for val in price_arr[cur_pos-CHART_TYPE_JDG_LEN:cur_pos]:
        if prev == -1.0:
            tmp_arr.append(0.0)
        else:
            tmp_arr.append(val - prev)
        prev = val

    return np.std(tmp_arr)

def get_macd(price_arr, cur_pos, period = 100):
    if cur_pos <= period:
        s = 0
    else:
        s = cur_pos - period
    tmp_arr = price_arr[s:cur_pos]
    tmp_arr.reverse()
    prices = np.array(tmp_arr, dtype=float)

    macd, macdsignal, macdhist = ta.MACD(prices,fastperiod=12, slowperiod=26, signalperiod=9)
    if macd[-1] > macdsignal[-1]:
        return 1
    else:
        return 0

# 日本時間で土曜7:00-月曜7:00までは取引不可として元データから取り除く
# なお、本来は月曜朝5:00から取引できるのが一般的なようである
def is_weekend(date_str):
    tz = pytz.timezone('Asia/Tokyo')
    dstr = date_str.replace(".","-")
    tdatetime = dt.strptime(dstr, '%Y-%m-%d %H:%M:%S')
    tz_time = tz.localize(tdatetime)
    gmt_plus2_tz = pytz.timezone('Etc/GMT+2')
    gmt_plus2_time = tz_time.astimezone(gmt_plus2_tz)
    week = gmt_plus2_time.weekday()
    return (week == 5 or week == 6)

def logfile_writeln_with_fd(out_fd, log_str):
    out_fd.write(log_str + "\n")
    out_fd.flush()

def make_serialized_data(start_idx, end_idx, step, x_arr_fpath, y_arr_fpath):
    input_mat = []
    angle_mat = []
    train_end_idx = -1
    print("all rate and data size: " + str(len(exchange_rates)))
    for i in range(start_idx, end_idx, step):
        if exchange_dates[i] == "2003-12-31 23:55:00":
            train_end_idx = i
        if i % 2000:
            print("current date idx: " + str(i))
        input_mat.append(
            [exchange_rates[i],
             (exchange_rates[i] - exchange_rates[i - 1]) / exchange_rates[i - 1],
             get_rsi(exchange_rates, i),
             get_ma(exchange_rates, i),
             get_ma_kairi(exchange_rates, i),
             get_bb_1(exchange_rates, i),
             get_bb_2(exchange_rates, i),
             get_ema(exchange_rates, i),
             get_cci(exchange_rates, i),
             get_mo(exchange_rates, i),
             get_vorarity(exchange_rates, i),
             get_macd(exchange_rates, i),
             judge_chart_type(exchange_rates[i - CHART_TYPE_JDG_LEN:i])
             ]
        )
        # input_mat.append(
        #     [reverse_exchange_rates[i],
        #      (reverse_exchange_rates[i] - reverse_exchange_rates[i - 1]) / reverse_exchange_rates[i - 1],
        #      get_rsi(reverse_exchange_rates, i),
        #      get_ma(reverse_exchange_rates, i),
        #      get_ma_kairi(reverse_exchange_rates, i),
        #      get_bb_1(reverse_exchange_rates, i),
        #      get_bb_2(reverse_exchange_rates, i),
        #      get_ema(reverse_exchange_rates, i),
        #      get_cci(reverse_exchange_rates, i),
        #      get_mo(reverse_exchange_rates, i),
        #      get_vorarity(reverse_exchange_rates, i),
        #      get_macd(reverse_exchange_rates, i),
        #      judge_chart_type(reverse_exchange_rates[i - CHART_TYPE_JDG_LEN:i])
        #      ]
        # )

        if y_arr_fpath != None:
            tmp = exchange_rates[i + PREDICT_FUTURE_LEGS] - exchange_rates[i]
            if tmp >= 0:
                angle_mat.append(1)
            else:
                angle_mat.append(0)

            # tmp = reverse_exchange_rates[i + PREDICT_FUTURE_LEGS] - reverse_exchange_rates[i]
            # if tmp >= 0:
            #     angle_mat.append(1)
            # else:
            #     angle_mat.append(0)

    with open(x_arr_fpath, 'wb') as f:
        pickle.dump(input_mat, f)
    with open(y_arr_fpath, 'wb') as f:
        pickle.dump(angle_mat, f)
    print("test data end index: " + str(train_end_idx))

    return input_mat, angle_mat

def setup_serialized_fx_data():
    global exchange_dates
    global exchange_rates
    global reverse_exchange_rates
    global tr_input_arr
    global tr_angle_arr
    global val_input_arr
    global val_angle_arr

    exchange_dates = []
    exchange_rates = []
    reverse_exchange_rates = []
    all_input_mat = []
    all_angle_mat = []
    tr_input_mat = []
    tr_angle_mat = []
    ts_input_mat = []

    if os.path.exists("./exchange_rates.pickle"):
        with open("./exchange_dates.pickle", 'rb') as f:
            exchange_dates = pickle.load(f)
        with open("./exchange_rates.pickle", 'rb') as f:
            exchange_rates = pickle.load(f)
    else:
        rates_fd = open('./USD_JPY_2001_2008_5min.csv', 'r')
        for line in rates_fd:
            splited = line.split(",")
            if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>" and splited[0] != "204/04/26" and splited[
                0] != "20004/04/26" and is_weekend(splited[0]) == False:
                time = splited[0].replace("/", "-")  # + " " + splited[1]
                val = float(splited[1])
                exchange_dates.append(time)
                exchange_rates.append(val)
        with open("./exchange_rates.pickle", 'wb') as f:
            pickle.dump(exchange_rates, f)
        with open("./exchange_dates.pickle", 'wb') as f:
            pickle.dump(exchange_dates, f)
    if os.path.exists("./all_input_mat.pickle"):
        with open('./all_input_mat.pickle', 'rb') as f:
            all_input_mat = pickle.load(f)
        with open('./all_angle_mat.pickle', 'rb') as f:
            all_angle_mat = pickle.load(f)
    else:
        # prev_org = -1
        # prev = -1
        # for rate in exchange_rates:
        #     if prev_org != -1:
        #         diff = rate - prev_org
        #         reverse_exchange_rates.append(prev - diff)
        #         prev_org = rate
        #         prev = prev - diff
        #     else:
        #         reverse_exchange_rates.append(rate)
        #         prev_org = rate
        #         prev = rate

        all_input_mat, all_angle_mat = \
            make_serialized_data(DATA_HEAD_ASOBI, len(exchange_rates) - DATA_HEAD_ASOBI - PREDICT_FUTURE_LEGS, SLIDE_IDX_NUM_AT_GEN_INPUTS_AND_COLLECT_LABELS, './all_input_mat.pickle', './all_angle_mat.pickle')

    tr_input_arr = np.array(all_input_mat[0:COMPETITION_TRAIN_DATA_NUM])
    tr_angle_arr = np.array(all_angle_mat[0:COMPETITION_TRAIN_DATA_NUM])
    ts_input_arr = np.array(all_input_mat[COMPETITION_TRAIN_DATA_NUM:])

    print("data size of all rates for train and test: " + str(len(exchange_rates)))
    #print("num of rate datas for tarin: " + str(COMPETITION_TRAIN_DATA_NUM_AT_RATE_ARR))
    print("input features sets for tarin: " + str(COMPETITION_TRAIN_DATA_NUM))
    print("input features sets for test: " + str(len(ts_input_arr)))
    print("finished setup environment data.")

def run_backtest():
    data_len = len(exchange_rates)

    log_fd_bt = open("./backtest_log_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt", mode = "w")
    # inner logger function for backtest
    def logfile_writeln_bt(log_str):
        nonlocal log_fd_bt
        log_fd_bt.write(log_str + "\n")
        log_fd_bt.flush()

    logfile_writeln_bt("start backtest...")

    portfolio = 1000000
    LONG = "LONG"
    SHORT = "SHORT"
    NOT_HAVE = "NOT_HAVE"
    pos_kind = NOT_HAVE
    HALF_SPREAD = 0.0015
    positions = 0
    trade_val = -1
    pos_cont_count = 0
    won_pips = 0
    start = time.time()
    ts_input_mat = []
    ts_input_arr = None

    ts_input_arr = np.array(ts_input_mat)

    # TODO: agentに環境を提供するような形でバックテストを実装する
    for window_s in range(data_len - COMPETITION_TRAIN_DATA_NUM - PREDICT_FUTURE_LEGS):
        current_spot = COMPETITION_TRAIN_DATA_NUM + window_s + PREDICT_FUTURE_LEGS

        logfile_writeln_bt(a_log_str_line)
        a_log_str_line = "log," + str(window_s)

        if pos_kind != NOT_HAVE:
            if pos_kind == LONG:
                cur_portfo = positions * (exchange_rates[current_spot] - HALF_SPREAD)
                diff = (exchange_rates[current_spot] - HALF_SPREAD) - trade_val
            elif pos_kind == SHORT:
                cur_portfo = portfolio + (positions * trade_val - positions * (exchange_rates[current_spot] + HALF_SPREAD))
                diff = trade_val - (exchange_rates[current_spot] + HALF_SPREAD)

    logfile_writeln_bt("finished backtest.")
    print("finished backtest.")
    process_time = time.time() - start
    logfile_writeln_bt("excecution time of backtest: " + str(process_time))
    logfile_writeln_bt("result of portfolio: " + str(portfolio))
    print("result of portfolio: " + str(portfolio))
    log_fd_bt.flush()
    log_fd_bt.close()
    return portfolio

# TODO:クラスとして利用できるようにまとめないといけない
#      get_env(学習用 or 評価用) ってな感じでenvを得られるように
# def run_script(mode):
#     if mode == "GEN_PICKLES":
#         setup_serialized_fx_data()
#     elif mode == "BACKTEST":
#         run_backtest()
#     else:
#         raise Exception(str(mode) + " mode is invalid.")
#
# if __name__ == '__main__':
#     if len(sys.argv) == 1:
#         run_script("TRAIN")
#         run_script("TRADE")

if __name__ == '__main__':
    setup_serialized_fx_data()
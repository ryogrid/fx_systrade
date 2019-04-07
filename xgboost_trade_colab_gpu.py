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
CHART_TYPE_JDG_LEN = 25

trade_log_fd = None

exchange_dates = None
exchange_rates = None
reverse_exchange_rates = None
is_use_gpu = False

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


def get_ema_rsi(price_arr, cur_pos, period = None):
    return 0

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

def get_lw(price_arr, cur_pos, period = None):
    return 0

def get_ss(price_arr, cur_pos, period = None):
    return 0

def get_dmi(price_arr, cur_pos, period = None):
    return 0

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

def is_weekend(date_str):
    tz = pytz.timezone('Asia/Tokyo')
    dstr = date_str.replace(".","-")
    tdatetime = dt.strptime(dstr, '%Y-%m-%d %H:%M:%S')
    tz_time = tz.localize(tdatetime)
    london_tz = pytz.timezone('Europe/London')
    london_time = tz_time.astimezone(london_tz)
    week = london_time.weekday()
    return (week == 5 or week == 6)

def logfile_writeln(log_str):
    trade_log_fd.write(log_str + "\n")
    trade_log_fd.flush()


def setup_historical_fx_data():
    global exchange_dates
    global exchange_rates
    global reverse_exchange_rates

    exchange_dates = []
    exchange_rates = []
    reverse_exchange_rates = []

    rates_fd = open('./USD_JPY_2001_2008_5min.csv', 'r')
    for line in rates_fd:
        splited = line.split(",")
        if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>"and splited[0] != "204/04/26" and splited[0] != "20004/04/26":
            time = splited[0].replace("/", "-") + " " + splited[1]
            val = float(splited[2])
            exchange_dates.append(time)
            exchange_rates.append(val)


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


def train_and_generate_model():
    data_len = len(exchange_rates)
    train_len = int(len(exchange_rates)/TRAINDATA_DIV)

    print("data size: " + str(data_len))
    print("train len: " + str(train_len))

    tr_input_mat = []
    tr_angle_mat = []
    for i in range(1000, train_len, OUTPUT_LEN):
        tr_input_mat.append(
            [exchange_rates[i],
             (exchange_rates[i] - exchange_rates[i - 1])/exchange_rates[i - 1],
             get_rsi(exchange_rates, i),
             get_ma(exchange_rates, i),
             get_ma_kairi(exchange_rates, i),
             get_bb_1(exchange_rates, i),
             get_bb_2(exchange_rates, i),
             get_ema(exchange_rates, i),
             get_ema_rsi(exchange_rates, i),
             get_cci(exchange_rates, i),
             get_mo(exchange_rates, i),
             get_lw(exchange_rates, i),
             get_ss(exchange_rates, i),
             get_dmi(exchange_rates, i),
             get_vorarity(exchange_rates, i),
             get_macd(exchange_rates, i),
             judge_chart_type(exchange_rates[i-CHART_TYPE_JDG_LEN:i])
         ]
            )
        tr_input_mat.append(
            [reverse_exchange_rates[i],
             (reverse_exchange_rates[i] - reverse_exchange_rates[i - 1])/reverse_exchange_rates[i - 1],
             get_rsi(reverse_exchange_rates, i),
             get_ma(reverse_exchange_rates, i),
             get_ma_kairi(reverse_exchange_rates, i),
             get_bb_1(reverse_exchange_rates, i),
             get_bb_2(reverse_exchange_rates, i),
             get_ema(reverse_exchange_rates, i),
             get_ema_rsi(reverse_exchange_rates, i),
             get_cci(reverse_exchange_rates, i),
             get_mo(reverse_exchange_rates, i),
             get_lw(reverse_exchange_rates, i),
             get_ss(reverse_exchange_rates, i),
             get_dmi(reverse_exchange_rates, i),
             get_vorarity(reverse_exchange_rates, i),
             get_macd(reverse_exchange_rates, i),
             judge_chart_type(reverse_exchange_rates[i-CHART_TYPE_JDG_LEN:i])
         ]
            )

        tmp = (exchange_rates[i+OUTPUT_LEN] - exchange_rates[i])/float(OUTPUT_LEN)
        if tmp >= 0:
            tr_angle_mat.append(1)
        else:
            tr_angle_mat.append(0)
        tmp = (reverse_exchange_rates[i+OUTPUT_LEN] - reverse_exchange_rates[i])/float(OUTPUT_LEN)
        if tmp >= 0:
            tr_angle_mat.append(1)
        else:
            tr_angle_mat.append(0)


    tr_input_arr = np.array(tr_input_mat)
    tr_angle_arr = np.array(tr_angle_mat)
    dtrain = xgb.DMatrix(tr_input_arr, label=tr_angle_arr)
    param = {'max_depth':6, 'eta':0.2, 'subsumble':0.5, 'silent':1, 'objective':'binary:logistic' }
    if is_use_gpu:
        param['updater'] = 'grow_gpu_hist'
        param['max_bin'] = 16
        param['gpu_id'] = 0
    else:
        param['nthread'] = 4

    watchlist  = [(dtrain,'train')]
    #num_round = 3000
    num_round = 100
    print("num_round: " + str(num_round))
    bst = xgb.train(param, dtrain, num_round, watchlist)

    bst.dump_model('./xgb_model.raw.txt')
    bst.save_model('./xgb.model')

    print("finished training and saved model.")

def run_backtest():
    global trade_log_fd

    print("start backtest...")

    data_len = len(exchange_rates)
    train_len = int(len(exchange_rates)/TRAINDATA_DIV)

    trade_log_fd = open("./backtest_log.txt", mode = "w")
    if is_use_gpu:
        bst = xgb.Booster({'predictor': 'gpu_predictor', 'tree_method': 'gpu_hist'})
    else:
        bst = xgb.Booster({'predictor': 'cpu_predictor', 'nthread': 4})

    bst.load_model("./xgb.model")
    portfolio = 1000000
    LONG = "LONG"
    SHORT = "SHORT"
    NOT_HAVE = "NOT_HAVE"
    pos_kind = NOT_HAVE
    HALF_SPREAD = 0.0015
    SONKIRI_RATE = 0.05
    RIKAKU_PIPS = 0.60

    positions = 0

    trade_val = -1

    pos_cont_count = 0
    won_pips = 0
    for window_s in range((data_len - train_len) - (OUTPUT_LEN)):
        current_spot = train_len + window_s + OUTPUT_LEN
        skip_flag = False

        # #rikaku
        # if pos_kind != NOT_HAVE:
        #     if pos_kind == LONG:
        #         got_pips = (exchange_rates[current_spot] - HALF_SPREAD) - trade_val
        #         cur_portfo = portfolio + (positions * (exchange_rates[current_spot] - HALF_SPREAD) - positions * trade_val)
        #     elif pos_kind == SHORT:
        #         got_pips = trade_val - (exchange_rates[current_spot] + HALF_SPREAD)
        #         cur_portfoo = portfolio + (positions * trade_val - positions * (exchange_rates[current_spot] + HALF_SPREAD))
        #     if got_pips >= RIKAKU_PIPS:
        #         portfolio = cur_portfo
        #         pos_kind = NOT_HAVE
        #         won_pips += got_pips
        #         print exchange_dates[current_spot] + " rikaku " + str(got_pips)
        #         continue

        if pos_kind != NOT_HAVE:
            if pos_kind == LONG:
                cur_portfo = positions * (exchange_rates[current_spot] - HALF_SPREAD)
                diff = (exchange_rates[current_spot] - HALF_SPREAD) - trade_val
            elif pos_kind == SHORT:
                cur_portfo = portfolio + (positions * trade_val - positions * (exchange_rates[current_spot] + HALF_SPREAD))
                diff = trade_val - (exchange_rates[current_spot] + HALF_SPREAD)
            if (cur_portfo - portfolio)/portfolio < -1*SONKIRI_RATE:
                portfolio = cur_portfo
                pos_kind = NOT_HAVE
                won_pips += diff
                logfile_writeln(str(diff) + "pips " + str(won_pips) + "pips")
                continue

        chart_type = judge_chart_type(exchange_rates[current_spot-CHART_TYPE_JDG_LEN:current_spot])
        if chart_type != 1 and chart_type != 2:
            skip_flag = True
            if pos_kind != NOT_HAVE:
                # if liner trend keep position
                continue


        if pos_kind != NOT_HAVE:
            if pos_cont_count >= (OUTPUT_LEN-1):
                if pos_kind == LONG:
                    pos_kind = NOT_HAVE
                    portfolio = positions * (exchange_rates[current_spot] - HALF_SPREAD)
                    diff = (exchange_rates[current_spot] - HALF_SPREAD) - trade_val
                    won_pips += diff
                    logfile_writeln(str(diff) + "pips " + str(won_pips) + "pips")
                    logfile_writeln(exchange_dates[current_spot] + " " + str(portfolio))
                elif pos_kind == SHORT:
                    pos_kind = NOT_HAVE
                    portfolio += positions * trade_val - positions * (exchange_rates[current_spot] + HALF_SPREAD)
                    diff = trade_val - (exchange_rates[current_spot] + HALF_SPREAD)
                    won_pips += diff
                    logfile_writeln(str(diff) + "pips " + str(won_pips) + "pips")
                    logfile_writeln(exchange_dates[current_spot] + " " + str(portfolio))
                pos_cont_count = 0
            else:
                pos_cont_count += 1
            continue

    #     vorarity = 0
        vorarity = get_vorarity(exchange_rates, current_spot)
        if vorarity >= 0.07:
            skip_flag = True

        # prediction
        ts_input_mat = []
        ts_input_mat.append(
           [exchange_rates[current_spot],
            (exchange_rates[current_spot] - exchange_rates[current_spot - 1])/exchange_rates[current_spot - 1],
            get_rsi(exchange_rates, current_spot),
            get_ma(exchange_rates, current_spot),
            get_ma_kairi(exchange_rates, current_spot),
            get_bb_1(exchange_rates, current_spot),
            get_bb_2(exchange_rates, current_spot),
            get_ema(exchange_rates, current_spot),
            get_ema_rsi(exchange_rates, current_spot),
            get_cci(exchange_rates, current_spot),
            get_mo(exchange_rates, current_spot),
            get_lw(exchange_rates, current_spot),
            get_ss(exchange_rates, current_spot),
            get_dmi(exchange_rates, current_spot),
            vorarity,
            get_macd(exchange_rates, current_spot),
            chart_type
        ]
        )

        ts_input_arr = np.array(ts_input_mat)
        dtest = xgb.DMatrix(ts_input_arr)

        pred = bst.predict(dtest)

        predicted_prob = pred[0]

        if pos_kind == NOT_HAVE and skip_flag == False:
            if predicted_prob >= 0.90 and chart_type == 2 :
               pos_kind = LONG
               positions = portfolio / (exchange_rates[current_spot] + HALF_SPREAD)
               trade_val = exchange_rates[current_spot] + HALF_SPREAD
            elif predicted_prob <= 0.1 and chart_type == 1:
               pos_kind = SHORT
               positions = portfolio / (exchange_rates[current_spot] - HALF_SPREAD)
               trade_val = exchange_rates[current_spot] - HALF_SPREAD

    print("finished backtest.")

def run_script(mode):
    global is_use_gpu

    if mode == "TRAIN":
        if exchange_dates == None:
            setup_historical_fx_data()
        train_and_generate_model()
    elif mode == "TRAIN_GPU":
        if exchange_dates == None:
            setup_historical_fx_data()
        is_use_gpu = True
        train_and_generate_model()
    elif mode == "TRADE":
        if exchange_dates == None:
            setup_historical_fx_data()
        run_backtest()
    else:
        raise Exception(str(mode) + " mode is invalid.")

if __name__ == '__main__':
    run_script("TRAIN")
    run_script("TRADE")

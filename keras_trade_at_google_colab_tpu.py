import numpy as np
import scipy.sparse
import pickle
import talib as ta
import os
import tensorflow as tf

#from tensorflow import keras

from tensorflow.keras.models import Sequential, , model_from_json
#from tensorflow.keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.layers import Dense, Dropout, Activation
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.layers.advanced_activations import PReLU
from tensorflow.keras.layers import PReLU
from keras.utils import np_utils, generic_utils

import tensorflow.keras.backend as K
from tensorflow.contrib.tpu.python.tpu import keras_support

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

INPUT_LEN = 1
OUTPUT_LEN = 5
TRAINDATA_DIV = 10
CHART_TYPE_JDG_LEN = 25

trade_log_fd = None

exchange_dates = None
exchange_rates = None
reverse_exchange_rates = None
#scaler = None

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
    #global scaler

    K.clear_session()

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

    X = np.array(tr_input_mat, dtype=np.float32)
    Y = np.array(tr_angle_mat, dtype=np.float32)

    X, scaler = preprocess_data(X)
    Y, encoder = preprocess_labels(Y)

    joblib.dump(scaler, "./sklearn.scaler.dump")

    np.random.seed(1337) # for reproducibility

    nb_classes = Y.shape[1]
    print(nb_classes, 'classes')

    dims = X.shape[1]
    print(dims, 'dims')

    neuro_num = 50

    # setup deep NN
    model = Sequential()
    model.add(Dense(neuro_num, input_shape=(dims,), activation="relu"))
    #model.add(Dense(neuro_num, activation="relu"))
    #model.add(BatchNormalization((neuro_num,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(int(neuro_num/2), activation="relu"))
    #model.add(BatchNormalization((neuro_num/2,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer="adam")

    # # TPU
    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    print("Training model...")
    model.fit(X, Y, nb_epoch=3000, batch_size=100, validation_split=0.15)

    dump_fd = open("./keras.model.json", "w")
    model_json_str = model.to_json()
    dump_fd.write(model_json_str)
    model.save_weights("./keras.weight")
    dump_fd.close()

def run_backtest():
    global trade_log_fd

    data_len = len(exchange_rates)
    train_len = int(len(exchange_rates)/TRAINDATA_DIV)

    # trade
    portfolio = 1000000
    LONG = "LONG"
    SHORT = "SHORT"
    NOT_HAVE = "NOT_HAVE"
    pos_kind = NOT_HAVE
    HALF_SPREAD = 0.0015
    SONKIRI_RATE = 0.05
    positions = 0
    trade_val = -1
    trade_log_fd = open("./backtest_log.txt", mode = "w")

    model_fd = open("./keras.model.json", "r")
    model = model_from_json(model_fd.read())
    model.load_weights("./keras.weight")
    scaler = joblib.load("./sklearn.scaler.dump")

    pos_cont_count = 0
    for window_s in range((data_len - train_len) - (OUTPUT_LEN)):
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
                    logfile_writeln("position close (LONG) " + exchange_dates[current_spot] + " " + str(portfolio))
                elif pos_kind == SHORT:
                    pos_kind = NOT_HAVE
                    portfolio += positions * trade_val - positions * (exchange_rates[current_spot] + HALF_SPREAD)
                    logfile_writeln("position close (SHORT) " + exchange_dates[current_spot] + " " + str(portfolio))
                pos_cont_count = 0
            else:
                pos_cont_count += 1
            continue

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

        X_test = np.array(ts_input_arr, dtype=np.float32)
        X_test, _ = preprocess_data(X_test, scaler)

        proba = model.predict_proba(X_test, verbose=0)

        logfile_writeln("state " + str(pos_kind))
        logfile_writeln("predicted probability -> UP: " + str(proba[0][0]) + ", DOWN: " + str(proba[0][1]))

        if pos_kind == NOT_HAVE and skip_flag == False:
            if proba[0][0] >= 0.9:
                pos_kind = LONG
                positions = portfolio / (exchange_rates[current_spot] + HALF_SPREAD)
                trade_val = exchange_rates[current_spot] + HALF_SPREAD
            elif proba[0][1] >= 0.9:
                pos_kind = SHORT
                positions = portfolio / (exchange_rates[current_spot] - HALF_SPREAD)
                trade_val = exchange_rates[current_spot] - HALF_SPREAD

        logfile_writeln("current portfolio " + exchange_dates[current_spot] + " " + str(portfolio))

def run_script(mode):
    if mode == "TRAIN":
        if exchange_dates == None:
            setup_historical_fx_data()
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

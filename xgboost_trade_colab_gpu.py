#!/usr/bin/python
import numpy as np
import scipy.sparse
import xgboost as xgb
import pickle
import talib as ta
from datetime import datetime as dt
import pytz
import os
import sys

import time
#from tensorboard_logger import configure, log_value

INPUT_LEN = 1
OUTPUT_LEN = 5
SLIDE_IDX_NUM_AT_GEN_INPUTS_AND_COLLECT_LABELS = 5
COMPETITION_DIV = True
COMPETITION_TRAIN_DATA_NUM = 208952
COMPETITION_TRAIN_DATA_NUM_AT_RATE_ARR = 522579

WHEN_TUNE_PARAM_THREAD_NUM = 1
RAPTOP_THREAD_NUM = 4
COLAB_CPU_AND_MBA_THREAD_NUM = 2
#THREAD_NUM = RAPTOP_THREAD_NUM

TRAINDATA_DIV = 2
CHART_TYPE_JDG_LEN = 25

VALIDATION_DATA_RATIO = 1.0 # rates of validation data to (all data - train data)
DATA_HEAD_ASOBI = 200

NUM_ROUND = 5000 #4000 #65 #4000
LONG_PROBA_THRESH = 0.8
SHORT_PROBA_THRESH = 0.2
VORARITY_THRESH = 0.1
ETA = 0.5
MAX_DEPTH = 5

FEATURE_NAMES = ["current_rate", "diff_ratio_between_previous_rate", "rsi", "ma", "ma_kairi", "bb_1", "bb_2", "ema", "ema_rsi", "cci", "mo", "lw", "ss", "dmi", "voratility", "macd", "chart_type"]
#FEATURE_NAMES = ["current_rate", "diff_ratio_between_previous_rate", "rsi", "ma", "ma_kairi", "bb_1", "bb_2", "ema", "mo", "voratility", "macd", "chart_type"]

OPTUNA_TRIAL_NUM = -1

#log_fd = None
log_fd_opt = None

tr_input_arr = None
tr_angle_arr = None
val_input_arr = None
val_angle_arr = None

exchange_dates = None
exchange_rates = None
reverse_exchange_rates = None
is_use_gpu = False
is_colab_cpu = False
is_param_tune_with_optuna = False
is_exec_at_mba = False

# if is_param_tune_with_optuna:
#     import optuna
#     from xgboost import XGBClassifier
#     from sklearn.metrics import accuracy_score

import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


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
    prev = -1.0
    for val in price_arr[cur_pos-CHART_TYPE_JDG_LEN:cur_pos]:
        if prev == -1.0:
            tmp_arr.append(0.0)
        else:
            tmp_arr.append(val - prev)
        prev = val

    return np.std(tmp_arr)

# def get_vorarity(price_arr, cur_pos, period = None):
#     tmp_arr = []
#     prev = -1
#     for val in price_arr[cur_pos-CHART_TYPE_JDG_LEN:cur_pos]:
#         if prev == -1:
#             tmp_arr.append(0)
#         else:
#             tmp_arr.append(val - prev)
#         prev = val
#
#     return np.std(tmp_arr)

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

# def logfile_writeln(log_str):
#     log_fd.write(log_str + "\n")
#     log_fd.flush()

def logfile_writeln_with_fd(out_fd, log_str):
    out_fd_opt.write(log_str + "\n")
    out_fd_opt.flush()

def logfile_writeln_opt(log_str):
    log_fd_opt.write(log_str + "\n")
    log_fd_opt.flush()

def set_tune_trial_num(tnum):
    global OPTUNA_TRIAL_NUM
    OPTUNA_TRIAL_NUM = tnum

def opt(trial):
    global LONG_PROBA_THRESH
    global SHORT_PROBA_THRESH
    global VORARITY_THRESH

    param = {}

    if is_use_gpu:
        param['tree_method'] = 'gpu_hist'
        param['max_bin'] = 16
        param['gpu_id'] = 0

    long_prob_thresh = trial.suggest_discrete_uniform('long_prob_thresh', 0.5, 0.9, 0.05)
    short_prob_thresh = trial.suggest_discrete_uniform('short_prob_thresh', 0.1, 0.5, 0.05)
    vorarity_thresh = trial.suggest_discrete_uniform('vorarity_thresh', 0.01, 0.3, 0.02)
    # LONG_PROBA_THRESH = long_prob_thresh
    # SHORT_PROBA_THRESH = short_prob_thresh
    # VORARITY_THRESH = vorarity_thresh

    eta = trial.suggest_discrete_uniform('eta', 0.05, 0.5, 0.05)
    n_estimators = trial.suggest_int('n_estimators', 0, 10000)
    #n_estimators = trial.suggest_int('n_estimators', 0, 100)
    max_depth = trial.suggest_int('max_depth', 1, 10)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)
    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)

    xgboost_tuna = XGBClassifier(
        max_depth = max_depth,
        random_state=42,
        n_estimators = n_estimators,
        min_child_weight = min_child_weight,
        subsample = subsample, # 0.7,
        colsample_bytree = colsample_bytree, # 0.6,
        eta = eta,
        objective = 'binary:logistic',
        verbosity = 0,
        n_thread = WHEN_TUNE_PARAM_THREAD_NUM,
        **param
    )

    verbosity = True
    if is_use_gpu or is_colab_cpu:
        verbosity = False
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        optuna.logging.disable_default_handler()
    xgboost_tuna.fit(tr_input_arr, tr_angle_arr, verbose=verbosity)
    booster = xgboost_tuna.get_booster()

    cur_params = {'long_prob_thresh':str(long_prob_thresh), 'short_prob_thresh':str(short_prob_thresh), 'vorarity_thresh':str(vorarity_thresh), 'eta':str(eta),
    'n_estimators':str(n_estimators), 'max_depth':str(max_depth), 'min_child_weight':str(min_child_weight),'subsample':str(subsample),'colsample_bytree':str(colsample_bytree)}
    logfile_writeln_opt(str(cur_params))
    portfolio_rslt = run_backtest(booster = booster, long_prob_thresh = long_prob_thresh, short_prob_thresh = short_prob_thresh, vorarity_thresh = vorarity_thresh)
    logfile_writeln_opt("portfolio_rslt =" + str(portfolio_rslt))
    #tuna_pred_test = xgboost_tuna.predict(val_input_arr)
    #return (1.0 - (accuracy_score(val_angle_arr, tuna_pred_test)))
    return (1.0 - ((portfolio_rslt/1000000.0) - 0.5))

def create_feature_map():
    outfile = open('fx_systrade_xgb.fmap', 'w')
    counter = 0
    for feat in FEATURE_NAMES:
        outfile.write('{0}\t{1}\tq\n'.format(counter, feat))
        counter += 1
    outfile.close()

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
    #global log_fd
    global log_fd_opt
    global tr_input_arr
    global tr_angle_arr
    global val_input_arr
    global val_angle_arr

    data_len = len(exchange_rates)
    # if is_param_tune_with_optuna:
    #     train_len = len(exchange_rates) - 1000 - OUTPUT_LEN
    # else:
    #     train_len = int(len(exchange_rates)/TRAINDATA_DIV)

    log_fd_tr = open("./train_progress_log_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt", mode = "w")
    # inner logger function for backtest
    def logfile_writeln_tr(log_str):
        nonlocal log_fd_tr
        log_fd_tr.write(log_str + "\n")
        log_fd_tr.flush()

    print("data size of rates: " + str(data_len))
    print("num of rate datas for tarin: " + str(COMPETITION_TRAIN_DATA_NUM_AT_RATE_ARR))
    print("input features sets for tarin: " + str(COMPETITION_TRAIN_DATA_NUM))


    logfile_writeln_tr("data size of rates: " + str(data_len))
    logfile_writeln_tr("num of rate datas for tarin: " + str(COMPETITION_TRAIN_DATA_NUM_AT_RATE_ARR))

    tr_input_mat = []
    tr_angle_mat = []

    if os.path.exists("./tr_input_mat.pickle"):
        with open('./tr_input_mat.pickle', 'rb') as f:
            tr_input_mat = pickle.load(f)
        with open('./tr_angle_mat.pickle', 'rb') as f:
            tr_angle_mat = pickle.load(f)
    else:
        for i in range(DATA_HEAD_ASOBI, len(exchange_rates) - DATA_HEAD_ASOBI - OUTPUT_LEN, SLIDE_IDX_NUM_AT_GEN_INPUTS_AND_COLLECT_LABELS):
            # if "2006" in exchange_dates[i]:
            #     print(len(tr_input_mat))
            #     print(str(DATA_HEAD_ASOBI + i - 1))
            #     quit()
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
                 str(judge_chart_type(exchange_rates[i-CHART_TYPE_JDG_LEN:i]))
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
                 str(judge_chart_type(reverse_exchange_rates[i-CHART_TYPE_JDG_LEN:i]))
             ]
                )

            tmp = exchange_rates[i+OUTPUT_LEN] - exchange_rates[i]
            if tmp >= 0:
                tr_angle_mat.append(1)
            else:
                tr_angle_mat.append(0)
            tmp = reverse_exchange_rates[i+OUTPUT_LEN] - reverse_exchange_rates[i]
            if tmp >= 0:
                tr_angle_mat.append(1)
            else:
                tr_angle_mat.append(0)

        with open('tr_input_mat.pickle', 'wb') as f:
            pickle.dump(tr_input_mat, f)
        with open('tr_angle_mat.pickle', 'wb') as f:
            pickle.dump(tr_angle_mat, f)

    #log output for tensorboard
    #configure("logs/xgboost_trade_cpu_1")

    # if is_param_tune_with_optuna:
    #     gen_data_len = int(((len(exchange_rates)/TRAINDATA_DIV))/5.0)
    # else:
    #     gen_data_len = len(tr_input_mat)
    tr_input_arr = np.array(tr_input_mat[0:COMPETITION_TRAIN_DATA_NUM])
    tr_angle_arr = np.array(tr_angle_mat[0:COMPETITION_TRAIN_DATA_NUM])
    dtrain = xgb.DMatrix(tr_input_arr, label=tr_angle_arr)

    split_idx = COMPETITION_TRAIN_DATA_NUM + int((len(tr_input_mat) - COMPETITION_TRAIN_DATA_NUM) * VALIDATION_DATA_RATIO)
    if VALIDATION_DATA_RATIO != 0.0:
        val_input_arr = np.array(tr_input_mat[COMPETITION_TRAIN_DATA_NUM:split_idx])
        val_angle_arr = np.array(tr_angle_mat[COMPETITION_TRAIN_DATA_NUM:split_idx])
        dval = xgb.DMatrix(val_input_arr, label=val_angle_arr)
        watchlist  = [(dtrain,'train'),(dval,'validation')]
    else:
        watchlist  = [(dtrain,'train')]

    start = time.time()
    if is_param_tune_with_optuna:
        # # backtestで log_fdが使われるので始末しておく
        # log_fd.flush()
        # log_fd.close()
        # log_fd = None

        log_fd_opt = open("./tune_progress_log_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt", mode = "w")
        study = optuna.create_study()
        #study = optuna.Study(study_name='distributed-example', storage='sqlite:///example.db')
        parallel_num = RAPTOP_THREAD_NUM * 2
        if is_colab_cpu or is_exec_at_mba:
            parallel_num = COLAB_CPU_AND_MBA_THREAD_NUM * 2
        study.optimize(opt, n_trials=OPTUNA_TRIAL_NUM, n_jobs=parallel_num)
        process_time = time.time() - start
        logfile_writeln_opt("best_params: " + str(study.best_params))
        logfile_writeln_opt("best_value: " + str(study.best_value))
        logfile_writeln_opt("best_trial: " + str(study.best_trial))
        logfile_writeln_opt("excecution time of tune: " + str(process_time))
        log_fd_opt.flush()
        log_fd_opt.close()
        quit()

    param = {'max_depth':MAX_DEPTH, 'eta':ETA, 'objective':'binary:logistic', 'verbosity':0, 'n_thread':RAPTOP_THREAD_NUM,'random_state':42, 'n_estimators':NUM_ROUND, 'min_child_weight': 15, 'subsample': 0.7, 'colsample_bytree':0.7}

    #param = {'max_depth':6, 'learning_rate':0.1, 'subsumble':0.5, 'objective':'binary:logistic', 'verbosity':0, 'booster': 'dart',
    # 'sample_type': 'uniform', 'normalize_type': 'tree', 'rate_drop': 0.1, 'skip_drop': 0.5}

    #param = {'max_depth':3, 'eta':0.1, 'objective':'binary:logistic', 'verbosity':0, 'n_thread':4,
    #    'random_state':42, 'n_estimators':NUM_ROUND, 'min_child_weight': 15, 'subsample': 0.7, 'colsample_bytree':0.7,
    #    'booster': 'dart', 'sample_type': 'uniform', 'normalize_type': 'tree', 'rate_drop': 0.1, 'skip_drop': 0.5}

    if is_use_gpu:
        param['tree_method'] = 'gpu_hist'
        param['max_bin'] = 16
        param['gpu_id'] = 0
        param['n_thread'] = COLAB_CPU_AND_MBA_THREAD_NUM
    if is_colab_cpu or is_exec_at_mba:
        param['n_thread'] = COLAB_CPU_AND_MBA_THREAD_NUM

    logfile_writeln_tr("training parameters are below...")
    logfile_writeln_tr(str(param))
    eval_result_dic = {}

    logfile_writeln_tr("num_round: " + str(NUM_ROUND))
    if is_colab_cpu or is_use_gpu:
        bst = xgb.train(param, dtrain, NUM_ROUND, evals=watchlist, evals_result=eval_result_dic, verbose_eval=False)
    else:
        bst = xgb.train(param, dtrain, NUM_ROUND, evals=watchlist, evals_result=eval_result_dic, verbose_eval=int(NUM_ROUND/100))
    process_time = time.time() - start
    logfile_writeln_tr("excecution time of training: " + str(process_time))

    # bst.dump_model('./xgb_model.raw.txt')
    # bst.save_model('./xgb.model')

    for ii in range(len(eval_result_dic['train']['error'])):
        if VALIDATION_DATA_RATIO != 0.0:
            logfile_writeln_tr(str(ii) + "," + str(eval_result_dic['train']['error'][ii]) + "," + str(eval_result_dic['validation']['error'][ii]))
        else:
            logfile_writeln_tr(str(ii) + "," + str(eval_result_dic['train']['error'][ii]))

    # feature importance
    create_feature_map()
    fti = bst.get_fscore(fmap='fx_systrade_xgb.fmap')
    logfile_writeln_tr('Feature Importances:')
    logfile_writeln_tr(str(fti))

    log_fd_tr.flush()
    log_fd_tr.close()

    print("finished training and saved model.")

def run_backtest(booster = None, long_prob_thresh = None, short_prob_thresh = None, vorarity_thresh = None):
    #global log_fd

    LONG_PROBA_THRESH_IN = LONG_PROBA_THRESH if long_prob_thresh == None else long_prob_thresh
    SHORT_PROBA_THRESH_IN = SHORT_PROBA_THRESH if short_prob_thresh == None else short_prob_thresh
    VORARITY_THRESH_IN = VORARITY_THRESH if vorarity_thresh == None else vorarity_thresh

    data_len = len(exchange_rates)
    #train_len = int(len(exchange_rates)/TRAINDATA_DIV)

    log_fd_bt = open("./backtest_log_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt", mode = "w")
    # inner logger function for backtest
    def logfile_writeln_bt(log_str):
        nonlocal log_fd_bt
        log_fd_bt.write(log_str + "\n")
        log_fd_bt.flush()

    logfile_writeln_bt("start backtest...")

    t_num = RAPTOP_THREAD_NUM
    if is_colab_cpu or is_exec_at_mba:
        t_num = COLAB_CPU_AND_MBA_THREAD_NUM
    if is_param_tune_with_optuna:
        t_num = WHEN_TUNE_PARAM_THREAD_NUM

    if booster == None:
        if is_use_gpu:
            bst = xgb.Booster({'predictor': 'gpu_predictor', 'tree_method': 'gpu_hist'})
        else:
            bst = xgb.Booster({'predictor': 'cpu_predictor', 'nthread': t_num})

        bst.load_model("./xgb.model")
    else:
        bst = booster #引数のものを使う
        bst.set_param({'nthread':t_num})

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
    start = time.time()
    ts_input_mat = []
    is_loaded_mat = False

    # if os.path.exists("./ts_input_mat.pickle"):
    #     with open('./ts_input_mat.pickle', 'rb') as f:
    #         ts_input_mat = pickle.load(f)
    #         is_loaded_mat = True

    logfile_writeln_bt("trade parameters LONG_PROBA_THRESH=" + str(LONG_PROBA_THRESH) + " SHORT_PROBA_THRESH=" + str(LONG_PROBA_THRESH) + " VORARITY_THRESH=" + str(VORARITY_THRESH) + " trade_trying_times=" + str(data_len - COMPETITION_TRAIN_DATA_NUM_AT_RATE_ARR - OUTPUT_LEN))
    # log format
    a_log_str_line = "log marker, loop count, Did Action == Sonkiri, chart_type, Did Action == skip according to chart_type, Did Action == Rieki Kakutei, Did Action == Skip according to position cointain time, voratility, Did Action == skip accordint to voratility, predicted prob, Get long position => 1 Get Short position => 2 else => 0, Did Action == Skip by chart_type at last decision"
    #logfile_writeln_bt("check_ts_input_mat,range func argument," + str(data_len - COMPETITION_TRAIN_DATA_NUM_AT_RATE_ARR - OUTPUT_LEN))
    #logfile_writeln_bt("check_ts_input_mat,current_sport start," + str(COMPETITION_TRAIN_DATA_NUM_AT_RATE_ARR + OUTPUT_LEN))
    for window_s in range(data_len - COMPETITION_TRAIN_DATA_NUM_AT_RATE_ARR - OUTPUT_LEN):
        #current_spot = DATA_HEAD_ASOBI + window_s # for trying backtest with trained period
        current_spot = COMPETITION_TRAIN_DATA_NUM_AT_RATE_ARR + window_s + OUTPUT_LEN

        logfile_writeln_bt(a_log_str_line)

        skip_flag = False
        delay_continue_flag = False
        vorarity = -1 # default value for log output
        a_log_str_line = "log," + str(window_s)

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
                logfile_writeln_bt(str(diff) + "pips " + str(won_pips) + "pips")
                a_log_str_line += ",1,0,0,0,0,0,0,0,0,0"
                #continue
                delay_continue_flag = True

        if delay_continue_flag == False:# or is_loaded_mat == False:
            chart_type = judge_chart_type(exchange_rates[current_spot-CHART_TYPE_JDG_LEN:current_spot])
            if chart_type != 1 and chart_type != 2:
                skip_flag = True
                if pos_kind != NOT_HAVE:
                    # if liner trend keep position
                    a_log_str_line += ",0," + str(chart_type) + ",1,0,0,0,0,0,0,0"
                    #continue
                    delay_continue_flag = True

        if pos_kind != NOT_HAVE and delay_continue_flag == False:
            if pos_cont_count >= (OUTPUT_LEN-1):
                if pos_kind == LONG:
                    pos_kind = NOT_HAVE
                    portfolio = positions * (exchange_rates[current_spot] - HALF_SPREAD)
                    diff = (exchange_rates[current_spot] - HALF_SPREAD) - trade_val
                    won_pips += diff
                    logfile_writeln_bt(str(diff) + "pips " + str(won_pips) + "pips")
                    logfile_writeln_bt(exchange_dates[current_spot] + " " + str(portfolio))
                    a_log_str_line += ",0," + str(chart_type) + ",0,1,0,0,0,0,0,0"
                elif pos_kind == SHORT:
                    pos_kind = NOT_HAVE
                    portfolio += positions * trade_val - positions * (exchange_rates[current_spot] + HALF_SPREAD)
                    diff = trade_val - (exchange_rates[current_spot] + HALF_SPREAD)
                    won_pips += diff
                    logfile_writeln_bt(str(diff) + "pips " + str(won_pips) + "pips")
                    logfile_writeln_bt(exchange_dates[current_spot] + " " + str(portfolio))
                    a_log_str_line += ",0," + str(chart_type) + ",0,1,0,0,0,0,0,0"
                pos_cont_count = 0
            else:
                a_log_str_line += ",0," + str(chart_type) + ",0,0,1,0,0,0,0,0"
                pos_cont_count += 1
            #continue
            delay_continue_flag = True

        if delay_continue_flag == False: #or is_loaded_mat == False:
            vorarity = get_vorarity(exchange_rates, current_spot)
#            if vorarity >= 0.07:
            if vorarity >= VORARITY_THRESH_IN:
                a_log_str_line += ",0," + str(chart_type) + ",0,0,0," + str(vorarity) + ",1,0,0,0"
                #continue
                delay_continue_flag = True

        if skip_flag and delay_continue_flag == False:
            a_log_str_line += ",0," + str(chart_type) + ",0,0,0," + str(vorarity) + ",0,0,0,1"
            #continue
            delay_continue_flag = True

        if delay_continue_flag == True:
            continue

        # prediction
        ts_input_mat = []
        if is_loaded_mat == False:
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
                str(chart_type)
            ]
            )
            #logfile_writeln_bt("check_ts_input_mat,check append window_s," + str(window_s) + "\n")

        ts_input_arr = np.array(ts_input_mat)
        dtest = xgb.DMatrix(ts_input_arr)

        pred = bst.predict(dtest)
        #print(pred)
        predicted_prob = pred[0]

        if pos_kind == NOT_HAVE and skip_flag == False:
            if predicted_prob > LONG_PROBA_THRESH_IN and chart_type == 2:
               pos_kind = LONG
               positions = portfolio / (exchange_rates[current_spot] + HALF_SPREAD)
               trade_val = exchange_rates[current_spot] + HALF_SPREAD
               a_log_str_line += ",0," + str(chart_type) + ",0,0,0," + str(vorarity) + ",1," + str(predicted_prob)  + ",1,0"
            elif predicted_prob < SHORT_PROBA_THRESH_IN and chart_type == 1:
               pos_kind = SHORT
               positions = portfolio / (exchange_rates[current_spot] - HALF_SPREAD)
               trade_val = exchange_rates[current_spot] - HALF_SPREAD
               a_log_str_line += ",0," + str(chart_type) + ",0,0,0," + str(vorarity) + ",1," + str(predicted_prob)  + ",2,0"
            else:
               a_log_str_line += ",0," + str(chart_type) + ",0,0,0," + str(vorarity) + ",1," + str(predicted_prob)  + ",0,0"
        else:
            raise Exception("this path should not be executed!!!!")
            #a_log_str_line += "0," + str(chart_type) + ",0,0,0," + str(vorarity) + ",1,0,0,1"

    # if is_loaded_mat == False:
    #     with open('./ts_input_mat.pickle', 'wb') as f:
    #         pickle.dump(ts_input_mat, f)

    logfile_writeln_bt("finished backtest.")
    process_time = time.time() - start
    logfile_writeln_bt("excecution time of backtest: " + str(process_time))
    log_fd_bt.flush()
    log_fd_bt.close()
    return portfolio

def run_script(mode):
    global is_use_gpu
    global is_colab_cpu
    global is_param_tune_with_optuna
    global THREAD_NUM
    global is_exec_at_mba

    if mode == "TRAIN":
        if exchange_dates == None:
            setup_historical_fx_data()
        train_and_generate_model()
    elif mode == "TRAIN_GPU":
        if exchange_dates == None:
            setup_historical_fx_data()
        is_use_gpu = True
        train_and_generate_model()
    elif mode == "TRAIN_COLAB_CPU":
        if exchange_dates == None:
            setup_historical_fx_data()
        is_colab_cpu = True
        train_and_generate_model()
    elif mode == "TRADE":
        if exchange_dates == None:
            setup_historical_fx_data()
        return run_backtest()
    elif mode == "TRADE_GPU":
        if exchange_dates == None:
            setup_historical_fx_data()
        is_use_gpu = True
        return run_backtest()
    elif mode == "TRADE_COLAB_CPU":
        if exchange_dates == None:
            setup_historical_fx_data()
        is_colab_cpu = True
        return run_backtest()
    elif mode == "CHANGE_TO_PARAM_TUNING_MODE":
        is_param_tune_with_optuna = True
    elif mode == "CHANGE_MBA_EXEC_MODE":
        is_exec_at_mba = True
    else:
        raise Exception(str(mode) + " mode is invalid.")

if __name__ == '__main__':
    # LONG_THRESH_CAND = [0.6, 0.65, 0.7, 0.75, 0.8]
    # SHORT_THRESH_CAND = [0.4, 0.35, 0.3, 0.25, 0.2]
    # ROUND_CAND = [4000, 5000, 6000, 7000, 8000]
    # with open("./my_search_result.txt", "w") as f:
    #     for ii in range(len(LONG_THRESH_CAND)):
    #         LONG_PROBA_THRESH = LONG_THRESH_CAND[ii]
    #         SHORT_PROBA_THRESH = SHORT_THRESH_CAND[ii]
    #         for jj in range(len(ROUND_CAND)):
    #             NUM_ROUND = ROUND_CAND[jj]
    #             run_script("TRAIN")
    #             result_portfolio = run_script("TRADE")
    #             f.write(str(LONG_PROBA_THRESH) + "," + str(SHORT_PROBA_THRESH) + "," + str(NUM_ROUND) + "," + str(result_portfolio) + "\n")

    # ETA_CAND = [0.05, 0.1, 0.3, 0.5]
    # MAX_DEPTH_CAND = [1, 3, 5, 7]
    # VORARITY_THRESH_CAND = [0.03, 0.07, 0.1]
    # with open("./my_search_result.txt", "w") as f:
    #     for ii in range(len(ETA_CAND)):
    #         ETA = ETA_CAND[ii]
    #         for jj in range(len(MAX_DEPTH_CAND)):
    #             MAX_DEPTH = MAX_DEPTH_CAND[jj]
    #             for kk in range(len(VORARITY_THRESH_CAND)):
    #                 VORARITY_THRESH = VORARITY_THRESH_CAND[kk]
    #                 run_script("TRAIN")
    #                 result_portfolio = run_script("TRADE")
    #                 f.write(str(ETA) + "," + str(MAX_DEPTH) + "," + str(VORARITY_THRESH) + "," + str(result_portfolio) + "\n")

    if len(sys.argv) == 1:
        run_script("TRAIN")
        run_script("TRADE")
    else:
        if sys.argv[1] == "--param-tune-win-raptop" or sys.argv[1] == "--param-tune-mac" or sys.argv[1] == "--param-tune-colab":
            if len(sys.argv) != 3:
                raise Exception("argment num is wrong.")

            run_script("CHANGE_TO_PARAM_TUNING_MODE")
            set_tune_trial_num(int(sys.argv[2]))

            if sys.argv[1] == "--param-tune-mac":
                is_exec_at_mba = True
            elif sys.argv[1] == "--param-tune-colab":
                is_colab_cpu = True

            run_script("TRAIN")
        else:
            raise Exception(sys.argv[1] + " is unknown argment.")

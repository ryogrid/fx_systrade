#!/usr/bin/python
import numpy as np
import scipy.sparse
import xgboost as xgb
import pickle
import talib as ta
import oandapy
from datetime import datetime
from time import sleep
import oanda_acount_info


INPUT_LEN = 1
OUTPUT_LEN = 5
TRAINDATA_DIV = 10
CHART_TYPE_JDG_LEN = 25

POSITION_UNITS =15000
SONKIRI_PIPS = 5 # convert to pips -> x100

oanda = oandapy.API(environment="live", access_token=oanda_acount_info.ACCESS_TOKEN)

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

def get_price_bid():

    try:
        response = oanda.get_prices(instruments="USD_JPY")
        prices = response.get("prices")
        return prices[0].get("bid")
    except:
        return -1

def get_price_ask():
    try:
        response = oanda.get_prices(instruments="USD_JPY")
        prices = response.get("prices")    
        return prices[0].get("ask")
    except:
        return -1
    
def exec_order_buy(cur_price):
    oanda.create_order(oanda_acount_info.ACOUNT_NUM,
                                  instrument="USD_JPY",
                                  units=POSITION_UNITS,
                                  side='buy',
                                  type='market',
                                  stopLoss= (cur_price-SONKIRI_PIPS)
                              )

def exec_order_sell(cur_price):
    oanda.create_order(oanda_acount_info.ACOUNT_NUM,
                                  instrument="USD_JPY",
                                  units=POSITION_UNITS,
                                  side='sell',
                                  type='market',
                                  stopLoss= (cur_price+SONKIRI_PIPS)
                              )    

def close_all_positions():
    oanda.close_position(oanda_acount_info.ACOUNT_NUM,
                         instrument="USD_JPY",
    )
    
"""
main
"""
from logging import getLogger,FileHandler,DEBUG,INFO

logger = getLogger(__name__)
_fhandler = FileHandler("./log/xgboost_oanda_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log",'w')
_fhandler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(_fhandler)

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


if True:
    bst = xgb.Booster({'nthread':4})
    bst.load_model("./hoge.model") 

if False: ### training start
    tr_input_mat = []
    tr_angle_mat = []
    for i in xrange(1000, train_len, OUTPUT_LEN):
        tr_input_mat.append(
            [exchange_rates[i],
             (exchange_rates[i] - exchange_rates[i - 1])/exchange_rates[i - 1],
#             (exchange_rates[i] - exchange_rates[i - OUTPUT_LEN])/float(OUTPUT_LEN),             
             get_rsi(exchange_rates, i),
             get_ma(exchange_rates, i),
             get_ma_kairi(exchange_rates, i),
             get_bb_1(exchange_rates, i),
             get_bb_2(exchange_rates, i),
             get_ema(exchange_rates, i),
             get_ema_rsi(exchange_rates, i),
             get_cci(exchange_rates, i),
             get_mo(exchange_rates, i),
#             get_po(exchange_rates, i),
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
#             (reverse_exchange_rates[i] - reverse_exchange_rates[i - OUTPUT_LEN])/float(OUTPUT_LEN),             
             get_rsi(reverse_exchange_rates, i),
             get_ma(reverse_exchange_rates, i),
             get_ma_kairi(reverse_exchange_rates, i),
             get_bb_1(reverse_exchange_rates, i),
             get_bb_2(reverse_exchange_rates, i),
             get_ema(reverse_exchange_rates, i),
             get_ema_rsi(reverse_exchange_rates, i),
             get_cci(reverse_exchange_rates, i),
             get_mo(reverse_exchange_rates, i),
#             get_po(reverse_exchange_rates, i),
             get_lw(reverse_exchange_rates, i),
             get_ss(reverse_exchange_rates, i),
             get_dmi(reverse_exchange_rates, i),
             get_vorarity(reverse_exchange_rates, i),
             get_macd(reverse_exchange_rates, i),
             judge_chart_type(reverse_exchange_rates[i-CHART_TYPE_JDG_LEN:i])             
         ]
            )        
#        print tr_input_mat

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

    watchlist  = [(dtrain,'train')]
    num_round = 3000 #10 #3000 # 1000
    bst = xgb.train(param, dtrain, num_round, watchlist)

    bst.dump_model('./dump.raw.txt')
    bst.save_model('./hoge.model')    
### training end

# trade
portfolio = 1000000
LONG = 1
SHORT = 2
NOT_HAVE = 3
pos_kind = NOT_HAVE
PRICES_LEN = 50


trade_val = -1

pos_cont_count = 0
oanda_prices_arr = [104.826, 104.856, 104.854, 104.837, 104.837, 104.81, 104.84, 104.813, 104.796, 104.787, 104.841, 104.918, 104.853, 104.821, 104.74, 104.74, 104.728, 104.738, 104.734, 104.706, 104.68, 104.674, 104.673, 104.664, 104.65, 104.679, 104.68, 104.705, 104.673, 104.672, 104.677, 104.613, 104.621, 104.625, 104.627, 104.641, 104.619, 104.613, 104.628, 104.623, 104.634, 104.622, 104.63, 104.622, 104.607, 104.599, 104.634, 104.63, 104.623, 104.622]
total_win_pips = 0
while 1:
    sleep(300) # 5min
    skip_flag = False
    
    latest_price_bid = get_price_bid()
    latest_price_ask = get_price_ask()
    # if API failed
    if latest_price_bid == -1 or latest_price_ask == -1:
        continue

    logger.debug(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " latest_price_bid " + str(latest_price_bid))
    oanda_prices_arr.insert(len(oanda_prices_arr), latest_price_bid)
    
    arr_len = len(oanda_prices_arr)
    if arr_len > PRICES_LEN:
        oanda_prices_arr.pop(0)
        logger.debug("arr" + str(oanda_prices_arr))
    elif arr_len == PRICES_LEN:
        print("arr has been filled!")
        logger.debug("arr has been filled!")
    elif arr_len < PRICES_LEN:
        continue

    #sonkiri
    if pos_kind != NOT_HAVE:
        if pos_kind == LONG:
            got_pips = latest_price_bid - trade_val
            cur_portfo = portfolio + (POSITION_UNITS * latest_price_bid - POSITION_UNITS * trade_val)
        elif pos_kind == SHORT:
            got_pips = trade_val - latest_price_ask
            cur_portfoo = portfolio + (POSITION_UNITS * trade_val - POSITION_UNITS * latest_price_ask)
        if got_pips < -1 * SONKIRI_PIPS:
            portfolio = cur_portfo
            pos_kind = NOT_HAVE
            close_all_positions()
            print datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " sonkiri " + str(got_pips)
            logger.debug(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " sonkiri " + str(got_pips))
            continue
    
    # chart_type = 0
    chart_type = judge_chart_type(oanda_prices_arr[-1*CHART_TYPE_JDG_LEN:-1])
    logger.debug("chart_type " + str(chart_type))
    if chart_type != 1 and chart_type != 2:
        skip_flag = True
        if pos_kind != NOT_HAVE:
            # if liner trend keep position
            pos_cont_count += 1 # this must not be skiped
            continue
        
        
    # print "state1 " + str(pos_kind)    
    if pos_kind != NOT_HAVE:
        # print "pos_cont_count " + str(pos_cont_count)
        if pos_cont_count >= (OUTPUT_LEN-1):
            if pos_kind == LONG:
                pos_kind = NOT_HAVE
                portfolio += (POSITION_UNITS * latest_price_bid) - POSITION_UNITS * trade_val
                print datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " " + str(portfolio)
                win_pips = latest_price_bid - trade_val
                total_win_pips += win_pips
                close_all_positions()                
                print datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " close " + str(win_pips) + " " + str(total_win_pips)
                logger.debug(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " close " + str(win_pips) + " " + str(total_win_pips))
            elif pos_kind == SHORT:
                pos_kind = NOT_HAVE
                portfolio += POSITION_UNITS * trade_val - POSITION_UNITS * latest_price_ask
                print datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " " + str(portfolio)
                win_pips = trade_val - latest_price_ask
                total_win_pips += win_pips
                close_all_positions()                
                print datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " close " + str(win_pips) + " " + str(total_win_pips)
                logger.debug(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " close " + str(win_pips) + " " + str(total_win_pips))
            pos_cont_count = 0
        else:
            pos_cont_count += 1
        continue

#    print("hoge")
    # try trade in only linear chart case

    # vorarity = 0
    vorarity = get_vorarity(oanda_prices_arr, -1)
    if vorarity >= 0.07:
        skip_flag = True
        logger.debug("skip because voraritiy")
        continue
#    print("vorarity: " + str(vorarity))
    
    # prediction    
    ts_input_mat = []
    ts_input_mat.append(
       [latest_price_bid,
        (oanda_prices_arr[-1] - oanda_prices_arr[-2])/oanda_prices_arr[-2],
#        (exchange_rates[current_spot] - exchange_rates[current_spot - OUTPUT_LEN])/float(OUTPUT_LEN),
        get_rsi(oanda_prices_arr, -1),
        get_ma(oanda_prices_arr, -1),
        get_ma_kairi(oanda_prices_arr, -1),
        get_bb_1(oanda_prices_arr, -1),
        get_bb_2(oanda_prices_arr, -1),
        get_ema(oanda_prices_arr, -1),
        get_ema_rsi(oanda_prices_arr, -1),
        get_cci(oanda_prices_arr, -1),
        get_mo(oanda_prices_arr, -1),
#        get_po(exchange_rates, current_spot),
        get_lw(oanda_prices_arr, -1),
        get_ss(oanda_prices_arr, -1),
        get_dmi(oanda_prices_arr, -1),
        vorarity,
        get_macd(oanda_prices_arr, -1),
        chart_type
    ]        
    )
#    print("vorarity: " + str(get_vorarity(exchange_rates, current_spot)))

    ts_input_arr = np.array(ts_input_mat)
    dtest = xgb.DMatrix(ts_input_arr)

    pred = bst.predict(dtest)

    predicted_prob = pred[0]

    # print "state2 " + str(pos_kind)
    # print "predicted_prob " + str(predicted_prob)
    # print "skip_flag:" + str(skip_flag)
    if pos_kind == NOT_HAVE and skip_flag == False:
        if predicted_prob >= 0.9 and chart_type == 2:
           pos_kind = LONG
           trade_val = latest_price_ask
           exec_order_buy(latest_price_ask)
           print datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " order buy " + str(latest_price_ask)
           logger.debug(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " order buy " + str(latest_price_ask))
        elif predicted_prob <= 0.1 and chart_type == 1:
           pos_kind = SHORT
           trade_val = latest_price_bid
           exec_order_sell(latest_price_bid)
           print datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " order sell " + str(latest_price_bid)
           logger.debug(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " order sell " + str(latest_price_bid))


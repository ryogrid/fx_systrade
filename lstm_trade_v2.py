#!/usr/bin/python
import numpy as np
import scipy.sparse
import pickle
from datetime import datetime as dt

OUTPUT_LEN = 5
TRAINDATA_DIV = 2



"""
main
"""
rates_fd = open('./hoge.csv', 'r')

exchange_dates = []
exchange_rates = []
for line in rates_fd:
    splited = line.split(",")
    if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>"and splited[0] != "204/04/26" and splited[0] != "20004/04/26": # and (not is_weekend(splited[0])):
        time = splited[0].replace("/", "-") + " " + splited[1]
#        val = float(splited[1])
        val = float(splited[2]) #for hoge.csv
        exchange_dates.append(time)
        exchange_rates.append(val)

data_len = len(exchange_rates)
train_len = len(exchange_rates)/TRAINDATA_DIV

print "data size: " + str(data_len)
print "train len: " + str(train_len)


if True: ### training start
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
    
    bst.dump_model('./dump7.raw.txt')
    bst.save_model('./hoge7.model')
    
### training end

# trade
portfolio = 1000000
LONG = 1
SHORT = 2
NOT_HAVE = 3
pos_kind = NOT_HAVE
HALF_SPREAD = 0.0015
SONKIRI_RATE = 0.05

positions = 0

trade_val = -1

pos_cont_count = 0
won_pips = 0
# q = ResultRate()
for window_s in xrange((data_len - train_len) - (OUTPUT_LEN)):
    current_spot = train_len + window_s + OUTPUT_LEN
    skip_flag = False

    if pos_kind != NOT_HAVE:
        if pos_kind == LONG:
            cur_portfo = positions * (exchange_rates[current_spot] - HALF_SPREAD)
            diff = (exchange_rates[current_spot] - HALF_SPREAD) - trade_val
        elif pos_kind == SHORT:
            cur_portfo = portfolio + (positions * trade_val - positions * (exchange_rates[current_spot] + HALF_SPREAD))
            diff = trade_val - (exchange_rates[current_spot] + HALF_SPREAD)
        if (cur_portfo - portfolio)/portfolio < -1*SONKIRI_RATE:
            # q.add_new_val(diff)
            portfolio = cur_portfo
            pos_kind = NOT_HAVE
            won_pips += diff
            print("loscat:" + str(diff*100) + "pips " + str(won_pips*100) + "pips")
            continue
        
    # chart_type = 0
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
                cur_portfo = positions * (exchange_rates[current_spot] - HALF_SPREAD)
                diff = (exchange_rates[current_spot] - HALF_SPREAD) - trade_val
                won_pips += diff
                print(str(diff*100) + "pips " + str(won_pips*100) + "pips")                
                print exchange_dates[current_spot] + " " + str(portfolio)
                portfolio = cur_portfo
            elif pos_kind == SHORT:
                pos_kind = NOT_HAVE
                cur_portfo = portfolio + positions * trade_val - positions * (exchange_rates[current_spot] + HALF_SPREAD)
                diff = trade_val - (exchange_rates[current_spot] + HALF_SPREAD)                
                won_pips += diff
                print(str(diff*100) + "pips " + str(won_pips*100) + "pips")                                
                print exchange_dates[current_spot] + " " + str(portfolio)
                portfolio = cur_portfo
            # q.add_new_val(diff)                
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
    dtest = xgb.DMatrix(ts_input_arr)

    pred = bst.predict(dtest)

    predicted_prob = pred[0]

    if pos_kind == NOT_HAVE and skip_flag == False:
        if predicted_prob >= 0.90 and chart_type == 2:
            # if q.is_tradable() == False:
            #     q.add_new_val(1)
            #     continue
            pos_kind = LONG
            positions = portfolio / (exchange_rates[current_spot] + HALF_SPREAD)
            trade_val = exchange_rates[current_spot] + HALF_SPREAD
        elif predicted_prob <= 0.1 and chart_type == 1:
            # if q.is_tradable() == False:
            #     q.add_new_val(1)
            #     continue            
            pos_kind = SHORT
            positions = portfolio / (exchange_rates[current_spot] - HALF_SPREAD)
            trade_val = exchange_rates[current_spot] - HALF_SPREAD

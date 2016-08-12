#!/usr/bin/python
from math import log

INIT_BALANCE = 250000
portfolio = INIT_BALANCE
positions = 0
all_trap_num = 0
balance = INIT_BALANCE
MARGIN_RATE = 0.04 #0.04
HALF_SPREAD = 0.02
BUY_LOTS = 600
WON_PIPS = 0.2 #0.3

UP = 1
DOWN = 2

def get_tuned_percent(start, end, cur_price, up_or_down):
#   return 1
    #return 1/((baseline_price/140)*2)
    #return (130 - (baseline_price - 90))/130
    #return (90 + (120 - baseline_price))/120.0

    tmp_end = end - start
    tmp_cur = cur_price - start
    center = tmp_end / 2.0

    if up_or_down == UP:
        sign = -1
    else: # DOWN
        sign = 1
        
    ret = (sign * 0.5 * (tmp_cur - center) + center) / float(center)

#    print("tuned percent " + str(ret))
    
    return ret


def get_baseline_lots(portfolio, cur_price):
    return BUY_LOTS
#     buyable_pos = (portfolio / MARGIN_RATE) * 0.5
#     left_traps = all_trap_num - positions

#     ret = int((buyable_pos / left_traps) / cur_price)
# #    print("lots " + str(ret))

#     return ret

def load_data(filepath):
    #rates_fd = open('./hoge.csv', 'r')
    rates_fd = open(filepath, 'r')
    exchange_dates = []
    exchange_rates = []
    for line in rates_fd:
        splited = line.split(",")
        if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>"and splited[0] != "204/04/26" and splited[0] != "20004/04/26":
            time = splited[0].replace("/", "-") + " " + splited[1]
            val = float(splited[1])
            #        val = float(splited[2]) # for hoge.csv
            exchange_dates.append(time)
            exchange_rates.append(val)
    return exchange_rates, exchange_dates

def make_trap(start, end, step):
    traps = []
    for price in xrange(100*start, 100*end, int(100*step)):
        traps.append([price/100.0, False, False, 0])

    return traps

def do_trade(currency_str, exchange_rates, cur, traps, up_or_down, last_positions, pos_limit, start, end):
    global balance
    print("current price " + currency_str + " = " + str(exchange_rates[cur]))
    
    #if no position, buy it
    for idx in xrange(len(traps)):
        if ((traps[idx][0] > (exchange_rates[cur-1]+HALF_SPREAD) \
           and traps[idx][0] <= (exchange_rates[cur]+HALF_SPREAD)) \
           or (traps[idx][0] > (exchange_rates[cur]+HALF_SPREAD) \
           and traps[idx][0] <= (exchange_rates[cur-1]+HALF_SPREAD))) \
           and traps[idx][1] == False \
           and last_positions <= pos_limit:
            traps[idx][1] = True
            traps[idx][3] = exchange_rates[cur]

    sign = 1 if up_or_down == UP else -1
    
    # close position
    for idx in xrange(len(traps)):
        if traps[idx][1] == True:
            if sign * ((exchange_rates[cur]-HALF_SPREAD) - traps[idx][3]) > WON_PIPS:
                balance += sign * ((exchange_rates[cur]-HALF_SPREAD) - traps[idx][3]) \
                           * get_baseline_lots(balance, traps[idx][3]) \
                           * get_tuned_percent(start, end, traps[idx][3], up_or_down)
                traps[idx][1] = False
                traps[idx][2] = False
                traps[idx][3] = 0

    margin_used = 0
    profit_or_loss = 0
    positions = 0
    for idx in xrange(len(traps)):
        if traps[idx][1] == True:
            margin_used += (traps[idx][3] *\
                            get_baseline_lots(balance, traps[idx][3]) \
                              * get_tuned_percent(start, end, traps[idx][3], up_or_down)) * MARGIN_RATE            
            profit_or_loss += sign * ((exchange_rates[cur]-HALF_SPREAD) - traps[idx][3]) \
                              * get_baseline_lots(balance, traps[idx][3]) \
                              * get_tuned_percent(start, end, traps[idx][3], up_or_down)
            positions += 1

    print(str(positions) + " "  + str(profit_or_loss))
    
    return margin_used, profit_or_loss, positions
    
"""
main
"""
exchange_rates1, exchange_dates1 = load_data('./USDJPY_UTC_1 Min_Bid_2010.05.10_2016.07.31.csv')
exchange_rates2, exchange_dates2 = load_data('./EURJPY_UTC_1 Min_Bid_2010.05.10_2016.07.31.csv')
exchange_rates3, exchange_dates3 = load_data('./TRYJPY_UTC_1 Min_Bid_2010.05.10_2016.07.31.csv')
exchange_rates4, exchange_dates4 = load_data('./NZDJPY_UTC_1 Min_Bid_2010.05.10_2016.07.31.csv')
exchange_rates5, exchange_dates5 = load_data('./AUDJPY_UTC_1 Min_Bid_2010.05.10_2016.07.31.csv')

data_len = len(exchange_rates1)

print "data size: " + str(data_len)

traps1 = make_trap(100, 120, 0.2)
traps2 = make_trap(100, 120, 0.2)
traps3 = make_trap(30, 50, 0.2)
traps4 = make_trap(60, 80, 0.2)
traps5 = make_trap(70, 90, 0.2)

all_trap_num = len(traps1) + len(traps2) + len(traps3) + len(traps4) + len(traps5)

#for cur in xrange(960000, data_len):
positions1 = 0
positions2 = 0
positions3 = 0
positions4 = 0
positions5 = 0
for cur in xrange(2757061, data_len):
    margin_used1, profit_or_loss1, positions1 = do_trade("USDJPY", exchange_rates1, cur, traps1, DOWN, positions1, 10, 100, 120)
    margin_used2, profit_or_loss2, positions2 = do_trade("EURJPY", exchange_rates2, cur, traps2, DOWN, positions2, 10, 100, 120)
    margin_used3, profit_or_loss3, positions3 = do_trade("TRYJPY", exchange_rates3, cur, traps3, UP, positions3, 10, 30, 50)
    margin_used4, profit_or_loss4, positions4 = do_trade("NZDJPY", exchange_rates4, cur, traps4, UP, positions4, 10, 60, 80)
    margin_used5, profit_or_loss5, positions5 = do_trade("AUDJPY", exchange_rates5, cur, traps5, UP, positions5, 10, 70, 90)

    positions = positions1 + positions2 + positions3 + positions4 + positions5
    profit_or_loss = profit_or_loss1 + profit_or_loss2 + profit_or_loss3 + profit_or_loss4 + profit_or_loss5
    margin_used = margin_used1 + margin_used2 + margin_used3 + margin_used4 + margin_used5
    portfolio = balance + profit_or_loss - margin_used

    if portfolio < margin_used * 0.2:
        print exchange_dates1[cur] + " " + str(positions) + " " + str(margin_used) + " " + str(portfolio) + " " + str(balance)
        print "dead"
        break

    print exchange_dates1[cur] + " " + str(positions) + " " +  str(margin_used) + " " + str(portfolio) + " " + str(balance)

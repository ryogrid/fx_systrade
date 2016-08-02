#!/usr/bin/python
from math import log

INIT_BALANCE = 500000
balance = INIT_BALANCE
MARGIN_RATE = 0.04
HALF_SPREAD = 0.015 
BUY_LOTS = 900
WON_PIPS = 0.3

def get_tuned_percent(baseline_price):
    #return 1
    #return 1/((baseline_price/140)*2)
    #return (130 - (baseline_price - 90))/130
    return (90 + (120 - baseline_price))/120.0

def get_tuned_percent2(baseline_price):
    #return 1
    #return 1/((baseline_price/140)*2)
    #return (130 - (baseline_price - 90))/130
    return baseline_price/180.0

def get_baseline_lots(portfolio, cur_price):
    return BUY_LOTS
#    return BUY_LOTS * (balance/INIT_BALANCE) * 0.3

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

"""
main
"""
exchange_rates, exchange_dates = load_data('./USDJPY_UTC_5 Mins_Bid_2003.08.03_2016.07.09.csv')
exchange_rates2, exchange_dates2 = load_data('./EURJPY_UTC_5 Mins_Bid_2003.08.03_2016.07.09.csv')

data_len = len(exchange_rates)

print "data size: " + str(data_len)

traps = []
traps2 = []

start = 90 # 80
end = 120 # 120
step = 0.2 # 0.1
for price in xrange(100*start, 100*end, int(100*step)):
    traps.append([price/100.0, False, False, 0])

start2 = 90 # 80
end2 = 120 # 120
step2 = 0.25 # 0.1
for price in xrange(100*start2, 100*end2, int(100*step2)):
    traps2.append([price/100.0, False, False, 0])    

positions = 0
positions2 = 0
#for cur in xrange(960000, data_len):
for cur in xrange(2, data_len):    
    print("current price1 = " + str(exchange_rates[cur]))
    print("current price2 = " + str(exchange_rates2[cur]))

    #if no position, buy it
    for idx in xrange(len(traps)):
        if ((traps[idx][0] > (exchange_rates[cur-1]+HALF_SPREAD) \
           and traps[idx][0] <= (exchange_rates[cur]+HALF_SPREAD)) \
           or (traps[idx][0] > (exchange_rates[cur]+HALF_SPREAD) \
           and traps[idx][0] <= (exchange_rates[cur-1]+HALF_SPREAD))) \
           and traps[idx][1] == False \
           and positions <= 30:
            traps[idx][1] = True
            traps[idx][3] = exchange_rates[cur]

    #if no position, buy it
    for idx in xrange(len(traps2)):
        if ((traps2[idx][0] > (exchange_rates2[cur-1]-HALF_SPREAD) \
           and traps2[idx][0] <= (exchange_rates2[cur]-HALF_SPREAD)) \
           or (traps2[idx][0] > (exchange_rates2[cur]-HALF_SPREAD) \
           and traps2[idx][0] <= (exchange_rates2[cur-1]-HALF_SPREAD))) \
           and traps2[idx][1] == False \
           and positions2 <= 30:
            traps2[idx][1] = True
            traps2[idx][3] = exchange_rates2[cur]

    # close position
    for idx in xrange(len(traps)):
        if traps[idx][1] == True:
            if (exchange_rates[cur]-HALF_SPREAD) - traps[idx][3] > WON_PIPS:
                balance += ((exchange_rates[cur]-HALF_SPREAD) - traps[idx][3]) \
                           * get_baseline_lots(balance, traps[idx][3]) \
                           * get_tuned_percent(traps[idx][3])
                traps[idx][1] = False
                traps[idx][2] = False
                traps[idx][3] = 0

    # close position
    for idx in xrange(len(traps2)):
        if traps2[idx][1] == True:
            if (traps2[idx][3] - exchange_rates2[cur]-HALF_SPREAD) > WON_PIPS:
                balance += (traps2[idx][3] - (exchange_rates2[cur] + HALF_SPREAD)) \
                           * get_baseline_lots(balance, traps2[idx][3]) \
                           * get_tuned_percent2(traps2[idx][3])
                traps2[idx][1] = False
                traps2[idx][2] = False
                traps2[idx][3] = 0

    margin_used = 0
    profit_or_loss1 = 0
    profit_or_loss2 = 0
    positions = 0
    positions2 = 0
    for idx in xrange(len(traps)):
        if traps[idx][1] == True:
            margin_used += (traps[idx][3] *\
                            get_baseline_lots(balance, traps[idx][3]) \
                              * get_tuned_percent(traps[idx][3])) * MARGIN_RATE            
            profit_or_loss1 += ((exchange_rates[cur]-HALF_SPREAD) - traps[idx][3]) \
                              * get_baseline_lots(balance, traps[idx][3]) \
                              * get_tuned_percent(traps[idx][3])
            positions += 1

    for idx in xrange(len(traps2)):
        if traps2[idx][1] == True:
            margin_used += (traps2[idx][3] *\
                            get_baseline_lots(balance, traps2[idx][3]) \
                              * get_tuned_percent2(traps2[idx][3])) * MARGIN_RATE
            profit_or_loss2 += (traps2[idx][3] - (exchange_rates2[cur]+HALF_SPREAD)) \
                              * get_baseline_lots(balance, traps2[idx][3]) \
                              * get_tuned_percent2(traps2[idx][3])
            positions2 += 1
            
    portfolio = balance + profit_or_loss1 + profit_or_loss2 - margin_used

    if portfolio < 0:
        print exchange_dates[cur] + " " + str(positions) + " " + str(margin_used) + " " + str(portfolio) + " " + str(balance)
        print "dead"
        break

    # print current status
#    if is_print == True:
    print exchange_dates[cur] + " " + str(positions) + " " + str(positions2) + " " + str(profit_or_loss1) +  " " + str(profit_or_loss2) + " " +  str(margin_used) + " " + str(portfolio) + " " + str(balance)

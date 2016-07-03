#!/usr/bin/python
from math import log

def get_tuned_percent(baseline_price):
    return log(150-baseline_price)

def get_baseline_lots(portfolio, cur_price):
    return portfolio / cur_price / 50
    
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


money = 5000000
portfolio = money
HALF_SPREAD = 0.015

BUY_LOTS = 500

WON_PIPS = 0.1
SONKIRI_PIPS = 0.5

center_val = float(int(exchange_rates[0]))
traps = []
for price in xrange(500):
    traps.append([center_val - price/10.0 , False, False, 5, 0])
    traps.append([center_val + price/10.0 , False, False, 5, 0])

prev_positions = 0
cur_positions = 0
for cur in xrange(1, data_len):
    is_print = False
    diff = exchange_rates[cur] - exchange_rates[cur-1]

    #if no position, buy it
    if diff >= 0:
        for idx in xrange(len(traps)):
            if traps[idx][0] > (exchange_rates[cur-1]+HALF_SPREAD) \
               and traps[idx][0] <= (exchange_rates[cur]+HALF_SPREAD) \
               and traps[idx][1] == False:
                traps[idx][1] = True
                traps[idx][4] = get_baseline_lots(portfolio, exchange_rates[cur]) * get_tuned_percent(exchange_rates[cur])
                money -= exchange_rates[cur] * get_baseline_lots(portfolio, exchange_rates[cur]) * get_tuned_percent(exchange_rates[cur])
                
    else:
        for idx in xrange(len(traps)):
            if traps[idx][0] < (exchange_rates[cur-1]+HALF_SPREAD) \
               and traps[idx][0] >= (exchange_rates[cur]+HALF_SPREAD) \
               and traps[idx][1] == False:
                traps[idx][1] = True
                traps[idx][4] = get_baseline_lots(portfolio, exchange_rates[cur]) * get_tuned_percent(exchange_rates[cur])                
                money -= exchange_rates[cur] * get_baseline_lots(portfolio, exchange_rates[cur]) * get_tuned_percent(exchange_rates[cur])

    # close position
    for idx in xrange(len(traps)):
        if traps[idx][1] == True:
            if (exchange_rates[cur]-HALF_SPREAD) - traps[idx][0] > WON_PIPS:
                if traps[idx][2] == False:
                    traps[idx][2] = True
                else:
                    traps[idx][2] = True # do nothing
                    
                    # money += (exchange_rates[cur]-HALF_SPREAD) * BUY_LOTS
                    # traps[idx][1] = False
                    # traps[idx][2] = False
                    # is_print = True
            elif traps[idx][2] == True and (exchange_rates[cur]-HALF_SPREAD) - traps[idx][0] < SONKIRI_PIPS:
                money += (exchange_rates[cur]-HALF_SPREAD) * traps[idx][4]
                traps[idx][1] = False
                traps[idx][2] = False
                is_print = True
           
    # # count positions
    # cur_positions = 0
    # for idx in xrange(len(traps)):
    #     if traps[idx][1] == True:
    #         cur_positions += 1
    
            
    # calculate portfolio
    if is_print == True:
        portfolio = money
        for idx in xrange(len(traps)):
            if traps[idx][1] == True:
                portfolio += (exchange_rates[cur] * traps[idx][4])
        print exchange_dates[cur] + " " + str(portfolio)



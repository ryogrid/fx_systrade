#!/usr/bin/python
from math import log
import oandapy
from datetime import datetime
from time import sleep
import oanda_acount_info
from logging import getLogger,FileHandler,DEBUG,INFO

POSITION_UNITS = 600
WON_PIPS = 0.3

UP = 1
DOWN = 2
all_trap_num = 0
positions_all = 0
portfolio = 0
MARGIN_RATE = 0.04

logger = getLogger(__name__)
_fhandler = FileHandler("./log/trap_repeat_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log",'w')
_fhandler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(_fhandler)

oanda = oandapy.API(environment="practice", access_token=oanda_acount_info.ACCESS_TOKEN)

def get_tuned_percent(baseline_price):
    return 1

def get_baseline_lots(portfolio, cur_price):
    return POSITION_UNITS
    # buyable_pos = (portfolio / MARGIN_RATE) * 0.8
    # left_traps = all_trap_num - positions_all

    # ret = int((buyable_pos / left_traps) / cur_price)

    # return ret
    
def get_price_bid(currency_str):

    try:
        response = oanda.get_prices(instruments=currency_str)
        prices = response.get("prices")
        return prices[0].get("bid")
    except:
        return -1

def get_price_ask(currency_str):
    try:
        response = oanda.get_prices(instruments=currency_str)
        prices = response.get("prices")    
        return prices[0].get("ask")
    except:
        return -1
    
def exec_order_buy(currency_str, cur_price, lots):
    responce = oanda.create_order(oanda_acount_info.ACOUNT_NUM,
                                  instrument=currency_str,
                                  units=lots,
                                  side='buy',
                                  type='market',
                                  takeProfit=cur_price+WON_PIPS)
#    return responce["tradeOpened"]["id"]

def exec_order_sell(currency_str, cur_price, lots):
    responce = oanda.create_order(oanda_acount_info.ACOUNT_NUM,
                                  instrument=currency_str,
                                  units=lots,
                                  side='sell',
                                  type='market',
                                  takeProfit=cur_price-WON_PIPS)
#    return responce["tradeOpened"]["id"]

def get_living_pos_list():
    return oanda.get_trades(oanda_acount_info.ACOUNT_NUM)
    
def make_trap(start, end, step):
    traps = []
    for price in xrange(100*start, 100*end, int(100*step)):
        traps.append([price/100.0, False, False, 0])

    return traps

def fill_trap(traps, currency_str, start, end, step, list_resp):
    prices_list = []
    for elem in list_resp["trades"]:
        if elem["instrument"] == currency_str:
            prices_list.append(elem["price"])
    print "prices_list " + currency_str + " " + str(prices_list)

    # fill open position infomation from server
    for price in prices_list:
        for idx in xrange(len(traps)):
            if (price >= traps[idx][0] \
                and price < (traps[idx][0]+step)) :
                traps[idx][1] = True
                traps[idx][3] = price
                print "recieve_idx" + str(idx)
                break

    return len(prices_list)

def get_yojo():
    return oanda.get_account(oanda_acount_info.ACOUNT_NUM)["marginAvail"]

def do_trade(currency_str, traps, up_or_down, pos_limit, step, server_pos_num):
    latest_price_bid = get_price_bid(currency_str)
    latest_price_ask = get_price_ask(currency_str)
    if latest_price_bid == -1 or latest_price_ask == -1:
        return -1
    
    if up_or_down == UP:
        price_open = latest_price_bid
        price_close = latest_price_ask
    else: # DOWN
        price_open = latest_price_ask
        price_close = latest_price_bid

    base_lots = get_baseline_lots(portfolio, price_close)
    
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " price_open " + str(price_open))

    positions = server_pos_num

    print("positions_from_server " + currency_str + " " + str(positions))
            
    #if no position, open it
    for idx in xrange(len(traps)):
        if (price_close >= traps[idx][0] \
            and price_close < (traps[idx][0] + step)) \
            and traps[idx][1] == False \
            and positions <= pos_limit:
            if up_or_down == UP:
                exec_order_buy(currency_str, price_open, base_lots)
            else:
                exec_order_sell(currency_str, price_open, base_lots)
            traps[idx][1] = True
            traps[idx][3] = price_open
            print "open_idx" + str(idx)
            positions += 1
            break

    sign = 1 if up_or_down == UP else -1
    
    profit_or_loss = 0
    for idx in xrange(len(traps)):
        if traps[idx][1] == True:
            profit_or_loss += sign * (price_close - traps[idx][3]) \
                              * base_lots \
                              * get_tuned_percent(traps[idx][3])

    print(str(positions) + " "  + str(profit_or_loss))
    
    return positions

                       
"""
main
"""
start1=95
end1=115
step1=0.2

start2=100
end2=120
step2=0.2

start3=30
end3=50
step3=0.2

start4=60
end4=80
step4=0.2

start5=70
end5=90
step5=0.2

# for all_trap_num
traps1 = make_trap(start1, end1, step1)
traps2 = make_trap(start2, end2, step2)
traps3 = make_trap(start3, end3, step3)
traps4 = make_trap(start4, end4, step4)
traps5 = make_trap(start5, end5, step5)    
all_trap_num = len(traps1) + len(traps2) + len(traps3) + len(traps4) + len(traps5)

while 1:
    sleep(15)

    portfolio = get_yojo()
    pos_list_resp = get_living_pos_list()
    
    traps1 = make_trap(start1, end1, step1)
    pos_num = fill_trap(traps1, "USD_JPY", start1, end1, step1, pos_list_resp)
    positions1 = do_trade("USD_JPY", traps1, DOWN, 10, step1, pos_num)

    traps2 = make_trap(start2, end2, step2)
    pos_num = fill_trap(traps2, "EUR_JPY", start2, end2, step2, pos_list_resp)
    positions2 = do_trade("EUR_JPY", traps2, DOWN, 10, step2, pos_num)

    traps3 = make_trap(start3, end3, step3)
    pos_num = fill_trap(traps3, "TRY_JPY", start3, end3, step3, pos_list_resp)    
    positions3 = do_trade("TRY_JPY", traps3, UP, 10, step3, pos_num)

    traps4 = make_trap(start4, end4, step4)    
    pos_num = fill_trap(traps4, "NZD_JPY", start4, end4, step4, pos_list_resp)    
    positions4 = do_trade("NZD_JPY", traps4, UP, 10, step4, pos_num)

    traps5 = make_trap(start5, end5, step5)
    pos_num = fill_trap(traps5, "AUD_JPY", start5, end5, step5, pos_list_resp)
    positions5 = do_trade("AUD_JPY", traps5, UP, 10, step5, pos_num)

    if positions1 == -1 or positions2 == -1 or positions3 == -1 or positions4 == -1 or positions5 == -1:
        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " api returns error")
        sleep(300) # 5min
        continue
    
    positions_all = positions1 + positions2 + positions3 + positions4 + positions5

    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " positions " + str(positions_all))    

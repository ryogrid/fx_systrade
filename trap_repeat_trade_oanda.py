#!/usr/bin/python
from math import log
import oandapy
from datetime import datetime
from time import sleep
import oanda_acount_info
from logging import getLogger,FileHandler,DEBUG,INFO

HALF_SPREAD = 0.015 
POSITION_UNITS = 800
WON_PIPS = 0.3

UP = 1
DOWN = 2

logger = getLogger(__name__)
_fhandler = FileHandler("./log/trap_repeat_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log",'w')
_fhandler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(_fhandler)

oanda = oandapy.API(environment="practice", access_token=oanda_acount_info.ACCESS_TOKEN)

def get_tuned_percent(baseline_price):
    return 1
    #return 1/((baseline_price/140)*2)
    #return (130 - (baseline_price - 90))/130
    #return (90 + (120 - baseline_price))/120.0

# def get_baseline_lots(portfolio, cur_price):
#     return BUY_LOTS
# #    return BUY_LOTS * (balance/INIT_BALANCE) * 0.9
# #     return BUY_LOTS * (1 + (((balance - INIT_BALANCE)/INIT_BALANCE) * 0.3))

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
    
def exec_order_buy(currency_str, cur_price):
    responce = oanda.create_order(oanda_acount_info.ACOUNT_NUM,
                                  instrument=currency_str,
                                  units=POSITION_UNITS,
                                  side='buy',
                                  type='market',
                                  takeProfit=cur_price+WON_PIPS)
    return responce["tradeOpened"]["id"]

def exec_order_sell(currency_str, cur_price):
    responce = oanda.create_order(oanda_acount_info.ACOUNT_NUM,
                                  instrument=currency_str,
                                  units=POSITION_UNITS,
                                  side='sell',
                                  type='market',
                                  takeProfit=cur_price-WON_PIPS)
    return responce["tradeOpened"]["id"]

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
    
def do_trade(currency_str, traps, up_or_down, pos_limit, step, server_pos_num):
    latest_price_bid = get_price_bid(currency_str)
    latest_price_ask = get_price_ask(currency_str)
    if latest_price_bid == -1 or latest_price_ask == -1:
        return
    
    if up_or_down == UP:
        price_open = latest_price_bid
        price_close = latest_price_ask
    else: # DOWN
        price_open = latest_price_ask
        price_close = latest_price_bid

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
                exec_order_buy(currency_str, price_open)
            else:
                exec_order_sell(currency_str, price_open)
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
                              * POSITION_UNITS \
                              * get_tuned_percent(traps[idx][3])

    print(str(positions) + " "  + str(profit_or_loss))
    
    return positions

                       
"""
main
"""
step1=0.5
step2=0.5
step3=0.5
step4=0.5
step5=0.5
while 1:
    sleep(15)

    pos_list_resp = get_living_pos_list()
    
    traps1 = make_trap(80, 120, step1)
    pos_num = fill_trap(traps1, "USD_JPY", 80, 120, step1, pos_list_resp)
    positions1 = do_trade("USD_JPY", traps1, UP, 10, step1, pos_num)

    traps2 = make_trap(90, 120, step2)
    pos_num = fill_trap(traps2, "EUR_JPY", 90, 130, step2, pos_list_resp)
    positions2 = do_trade("EUR_JPY", traps2, DOWN, 10, step2, pos_num)

    traps3 = make_trap(30, 60, step3)
    pos_num = fill_trap(traps3, "TRY_JPY", 20, 45, step3, pos_list_resp)    
    positions3 = do_trade("TRY_JPY", traps3, UP, 10, step3, pos_num)

    traps4 = make_trap(50, 90, step4)    
    pos_num = fill_trap(traps4, "NZD_JPY", 55, 90, step4, pos_list_resp)    
    positions4 = do_trade("NZD_JPY", traps4, DOWN, 10, step4, pos_num)

    traps5 = make_trap(60, 100, step5)
    pos_num = fill_trap(traps5, "AUD_JPY", 70, 100, step5, pos_list_resp)
    positions5 = do_trade("AUD_JPY", traps5, UP, 10, step5, pos_num)

    positions_all = positions1 + positions2 + positions3 + positions4 + positions5

    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " positions " + str(positions_all))    

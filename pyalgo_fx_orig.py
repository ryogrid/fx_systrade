from pyalgotrade import strategy
from pyalgotrade import plotter
from pyalgotrade.stratanalyzer import sharpe
from pyalgotrade.barfeed import yahoofeed
from pyalgotrade.barfeed import csvfeed
from pyalgotrade.barfeed import Frequency

from pyalgotrade.technical import ma
from pyalgotrade.technical import cross
import talib
from pyalgotrade.talibext.indicator import LINEARREG

import pandas.io.data as web
import pytz
from datetime import datetime


class MATrade(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument):
        strategy.BacktestingStrategy.__init__(self, feed)
        self.__instrument = instrument
        self.__position = None
        # We'll use adjusted close values instead of regular close values.
        self.setUseAdjustedValues(True)
        self.__prices = feed[instrument].getPriceDataSeries()
        self.__sma = ma.SMA(self.__prices, 20)
        self.__pos_kind = ""
        
    def getSMA(self):
        return self.__sma

    def onEnterCanceled(self, position):
        self.__position = None
    
    def onExitOk(self, position):
        self.__position = None

    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        self.__position.exitMarket()

    def touchTop(self):
        if cross.cross_above(self.__prices, self.__sma, -4, -2) > 0 and \
           cross.cross_below(self.__prices, self.__sma, -2) > 0:
            return True

    def touchBottom(self):
        if cross.cross_below(self.__prices, self.__sma, -4, -2) > 0 and \
           cross.cross_above(self.__prices, self.__sma, -2) > 0:
            return True        
        
    def onBars(self, bars):
        if self.__position != None and self.__position.getReturn() < -0.05:
            self.__position.exitMarket()
            
        # If a position was not opened, check if we should enter a long position.
        if self.__position is None:
            ds = self.getFeed().getDataSeries(self.__instrument).getCloseDataSeries()
            if self.touchBottom(): # and LINEARREG(ds, 20)[1] > 0:
                shares = int(self.getBroker().getCash() * 0.9 / bars[self.__instrument].getPrice())
                # Enter a buy market order. The order is good till canceled.
                self.__position = self.enterShort(self.__instrument, shares, True)
                self.__pos_kind = "Short"
                print str(bars[self.__instrument].getDateTime()) + " " + "buy Short"
            elif self.touchTop(): # and LINEARREG(ds, 20)[1] < 0:
                shares = int(self.getBroker().getCash() * 0.9 / bars[self.__instrument].getPrice())
                # Enter a buy market order. The order is good till canceled.
                self.__position = self.enterLong(self.__instrument, shares, True)
                self.__pos_kind = "Long"
                print str(bars[self.__instrument].getDateTime()) + " " + "buy Long"
            return
        # Check if we have to exit the position.
        elif (not self.__position.exitActive()):
            if self.__pos_kind == "Long" and (cross.cross_below(self.__prices, self.__sma, -2) > 0 or self.touchBottom()):
                self.__position.exitMarket() #Long exit
                print str(bars[self.__instrument].getDateTime()) + " " + "exit " + self.__pos_kind
            elif self.__pos_kind == "Short" and (cross.cross_above(self.__prices, self.__sma, -2) > 0 or self.touchTop()):
                self.__position.exitMarket() #Short exit
                print str(bars[self.__instrument].getDateTime()) + " " + "exit " + self.__pos_kind
                
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
    
def main(plot):
#    merge_csv("./hoge.csv", ["./USDJPY_M5_2001.txt","./USDJPY_M5_2002.txt","./USDJPY_M5_2003.txt","./USDJPY_M5_2004.txt",\
#                             "./USDJPY_M5_2005.txt","./USDJPY_M5_2006.txt","./USDJPY_M5_2007.txt","./USDJPY_M5_2008.txt"])

    instrument = "USDJPY"
    feed = csvfeed.GenericBarFeed(Frequency.MINUTE, pytz.utc)
    feed.addBarsFromCSV(instrument, "./hoge.csv")
    strat = MATrade(feed, instrument)
    sharpeRatioAnalyzer = sharpe.SharpeRatio()
    strat.attachAnalyzer(sharpeRatioAnalyzer)
    
    if plot:
        plt = plotter.StrategyPlotter(strat, True, True, True)
        plt.getInstrumentSubplot(instrument).addDataSeries("MA20", strat.getSMA())

    strat.run()
    print "Sharpe ratio: %.2f" % sharpeRatioAnalyzer.getSharpeRatio(0.05)

    if plot:
        plt.plot()

if __name__ == "__main__":
    main(True)

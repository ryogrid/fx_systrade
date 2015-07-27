from pyalgotrade import strategy
from pyalgotrade import plotter
from pyalgotrade.stratanalyzer import sharpe
from pyalgotrade.barfeed import yahoofeed
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
        # If a position was not opened, check if we should enter a long position.
        if self.__position is None:
            ds = self.getFeed().getDataSeries(self.__instrument).getCloseDataSeries()
            if self.touchBottom() and LINEARREG(ds, 20)[-1] > 0:
                shares = int(self.getBroker().getCash() * 0.9 / bars[self.__instrument].getPrice())
                # Enter a buy market order. The order is good till canceled.
                self.__position = self.enterLong(self.__instrument, shares, True)

            return
        # Check if we have to exit the position.
        elif not self.__position.exitActive() and \
            (self.touchTop() or cross.cross_above(self.__prices, self.__sma, -2) > 0 or self.touchBottom()):
            self.__position.exitMarket()


def main(plot):
    start = datetime(2000, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2015, 7, 1, 0, 0, 0, 0, pytz.utc)
    data = web.DataReader('DEXJPUS', 'fred', start, end)

    frslt = open('./hoge.csv', 'w')
    frslt.write("Date,Open,High,Low,Close,Volume,Adj Close\n")
    for k,v in data["DEXJPUS"].iteritems():
        if v != v: # if nan
            continue
            
        frslt.write(str(k) + "," + str(v) + "," + \
                    str(v) + "," + str(v) + \
                    "," + str(v) + ",1000000,"+ str(v) + "\n")
    frslt.close()

    instrument = "USDJPY"
    feed = yahoofeed.Feed()
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

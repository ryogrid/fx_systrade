from pyalgotrade import strategy
from pyalgotrade import plotter
from pyalgotrade.technical import bollinger
from pyalgotrade.stratanalyzer import sharpe
from pyalgotrade.barfeed import yahoofeed
import pandas.io.data as web
import pytz
from datetime import datetime


class BBands(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, bBandsPeriod):
        strategy.BacktestingStrategy.__init__(self, feed)
        self.__instrument = instrument
        self.__bbands = bollinger.BollingerBands(feed[instrument].getCloseDataSeries(), bBandsPeriod, 2)

    def getBollingerBands(self):
        return self.__bbands

    def onBars(self, bars):
        lower = self.__bbands.getLowerBand()[-1]
        upper = self.__bbands.getUpperBand()[-1]
        if lower is None:
            return

        shares = self.getBroker().getShares(self.__instrument)
        bar = bars[self.__instrument]
        if shares == 0 and bar.getClose() < lower:
#            sharesToBuy = int(self.getBroker().getCash(False) / bar.getClose())
            sharesToBuy = int(self.getBroker().getCash(False) / 200)
            self.marketOrder(self.__instrument, sharesToBuy)
        elif shares > 0 and bar.getClose() > upper:
            self.marketOrder(self.__instrument, -1*shares)

def main(plot):
    bBandsPeriod = 40

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
    strat = BBands(feed, instrument, bBandsPeriod)
    sharpeRatioAnalyzer = sharpe.SharpeRatio()
    strat.attachAnalyzer(sharpeRatioAnalyzer)
    
    if plot:
        plt = plotter.StrategyPlotter(strat, True, True, True)
        plt.getInstrumentSubplot(instrument).addDataSeries("upper", strat.getBollingerBands().getUpperBand())
        plt.getInstrumentSubplot(instrument).addDataSeries("middle", strat.getBollingerBands().getMiddleBand())
        plt.getInstrumentSubplot(instrument).addDataSeries("lower", strat.getBollingerBands().getLowerBand())

    strat.run()
    print "Sharpe ratio: %.2f" % sharpeRatioAnalyzer.getSharpeRatio(0.05)

    if plot:
        plt.plot()

if __name__ == "__main__":
    main(True)

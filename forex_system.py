# coding: utf-8
 
import matplotlib.pyplot as plt
import numpy as np
import oandapy
import pandas as pd
from datetime import datetime
from datetime import timedelta
from scipy import optimize
from scipy.optimize import differential_evolution
 
class ForexSystem(object):
    def __init__(self, environment=None, account_id=None, access_token=None):
        '''初期化する。
          Args:
              environment: 環境。
              account_id: アカウントID。
              access_token: アクセストークン。
        '''
        self.environment = environment
        self.account_id = account_id
        self.access_token = access_token
        # バックテストの場合（バックテストの場合、環境を指定しないので）
        if environment is None:
            # 1分足を格納する。
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/AUD_USD1.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.audusd1_op = temp.iloc[:, 0]
            self.audusd1_hi = temp.iloc[:, 1]
            self.audusd1_lo = temp.iloc[:, 2]
            self.audusd1_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/EUR_USD1.csv",index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.eurusd1_op = temp.iloc[:, 0]
            self.eurusd1_hi = temp.iloc[:, 1]
            self.eurusd1_lo = temp.iloc[:, 2]
            self.eurusd1_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/GBP_USD1.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.gbpusd1_op = temp.iloc[:, 0]
            self.gbpusd1_hi = temp.iloc[:, 1]
            self.gbpusd1_lo = temp.iloc[:, 2]
            self.gbpusd1_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/USD_JPY1.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.usdjpy1_op = temp.iloc[:, 0]
            self.usdjpy1_hi = temp.iloc[:, 1]
            self.usdjpy1_lo = temp.iloc[:, 2]
            self.usdjpy1_cl = temp.iloc[:, 3]
            # 5分足を格納する。
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/AUD_USD5.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.audusd5_op = temp.iloc[:, 0]
            self.audusd5_hi = temp.iloc[:, 1]
            self.audusd5_lo = temp.iloc[:, 2]
            self.audusd5_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/EUR_USD5.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.eurusd5_op = temp.iloc[:, 0]
            self.eurusd5_hi = temp.iloc[:, 1]
            self.eurusd5_lo = temp.iloc[:, 2]
            self.eurusd5_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/GBP_USD5.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.gbpusd5_op = temp.iloc[:, 0]
            self.gbpusd5_hi = temp.iloc[:, 1]
            self.gbpusd5_lo = temp.iloc[:, 2]
            self.gbpusd5_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/USD_JPY5.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.usdjpy5_op = temp.iloc[:, 0]
            self.usdjpy5_hi = temp.iloc[:, 1]
            self.usdjpy5_lo = temp.iloc[:, 2]
            self.usdjpy5_cl = temp.iloc[:, 3]
            # 15分足を格納する。
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/AUD_USD15.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.audusd15_op = temp.iloc[:, 0]
            self.audusd15_hi = temp.iloc[:, 1]
            self.audusd15_lo = temp.iloc[:, 2]
            self.audusd15_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/EUR_USD15.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.eurusd15_op = temp.iloc[:, 0]
            self.eurusd15_hi = temp.iloc[:, 1]
            self.eurusd15_lo = temp.iloc[:, 2]
            self.eurusd15_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/GBP_USD15.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.gbpusd15_op = temp.iloc[:, 0]
            self.gbpusd15_hi = temp.iloc[:, 1]
            self.gbpusd15_lo = temp.iloc[:, 2]
            self.gbpusd15_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/USD_JPY15.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.usdjpy15_op = temp.iloc[:, 0]
            self.usdjpy15_hi = temp.iloc[:, 1]
            self.usdjpy15_lo = temp.iloc[:, 2]
            self.usdjpy15_cl = temp.iloc[:, 3]
            # 30分足を格納する。
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/AUD_USD30.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.audusd30_op = temp.iloc[:, 0]
            self.audusd30_hi = temp.iloc[:, 1]
            self.audusd30_lo = temp.iloc[:, 2]
            self.audusd30_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/EUR_USD30.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.eurusd30_op = temp.iloc[:, 0]
            self.eurusd30_hi = temp.iloc[:, 1]
            self.eurusd30_lo = temp.iloc[:, 2]
            self.eurusd30_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/GBP_USD30.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.gbpusd30_op = temp.iloc[:, 0]
            self.gbpusd30_hi = temp.iloc[:, 1]
            self.gbpusd30_lo = temp.iloc[:, 2]
            self.gbpusd30_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/USD_JPY30.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.usdjpy30_op = temp.iloc[:, 0]
            self.usdjpy30_hi = temp.iloc[:, 1]
            self.usdjpy30_lo = temp.iloc[:, 2]
            self.usdjpy30_cl = temp.iloc[:, 3]
            # 1時間足を格納する。
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/AUD_USD60.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.audusd60_op = temp.iloc[:, 0]
            self.audusd60_hi = temp.iloc[:, 1]
            self.audusd60_lo = temp.iloc[:, 2]
            self.audusd60_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/EUR_USD60.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.eurusd60_op = temp.iloc[:, 0]
            self.eurusd60_hi = temp.iloc[:, 1]
            self.eurusd60_lo = temp.iloc[:, 2]
            self.eurusd60_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/GBP_USD60.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.gbpusd60_op = temp.iloc[:, 0]
            self.gbpusd60_hi = temp.iloc[:, 1]
            self.gbpusd60_lo = temp.iloc[:, 2]
            self.gbpusd60_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/USD_JPY60.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.usdjpy60_op = temp.iloc[:, 0]
            self.usdjpy60_hi = temp.iloc[:, 1]
            self.usdjpy60_lo = temp.iloc[:, 2]
            self.usdjpy60_cl = temp.iloc[:, 3]
            # 4時間足を格納する。
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/AUD_USD240.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.audusd240_op = temp.iloc[:, 0]
            self.audusd240_hi = temp.iloc[:, 1]
            self.audusd240_lo = temp.iloc[:, 2]
            self.audusd240_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/EUR_USD240.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.eurusd240_op = temp.iloc[:, 0]
            self.eurusd240_hi = temp.iloc[:, 1]
            self.eurusd240_lo = temp.iloc[:, 2]
            self.eurusd240_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/GBP_USD240.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.gbpusd240_op = temp.iloc[:, 0]
            self.gbpusd240_hi = temp.iloc[:, 1]
            self.gbpusd240_lo = temp.iloc[:, 2]
            self.gbpusd240_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/USD_JPY240.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.usdjpy240_op = temp.iloc[:, 0]
            self.usdjpy240_hi = temp.iloc[:, 1]
            self.usdjpy240_lo = temp.iloc[:, 2]
            self.usdjpy240_cl = temp.iloc[:, 3]
            # 日足を格納する。
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/AUD_USD1440.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.audusd1440_op = temp.iloc[:, 0]
            self.audusd1440_hi = temp.iloc[:, 1]
            self.audusd1440_lo = temp.iloc[:, 2]
            self.audusd1440_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/EUR_USD1440.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.eurusd1440_op = temp.iloc[:, 0]
            self.eurusd1440_hi = temp.iloc[:, 1]
            self.eurusd1440_lo = temp.iloc[:, 2]
            self.eurusd1440_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/GBP_USD1440.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.gbpusd1440_op = temp.iloc[:, 0]
            self.gbpusd1440_hi = temp.iloc[:, 1]
            self.gbpusd1440_lo = temp.iloc[:, 2]
            self.gbpusd1440_cl = temp.iloc[:, 3]
            temp = pd.read_csv(
                "~/work/etc/fx_systrade/historical_data/USD_JPY1440.csv", index_col=0, header=0)
            index = pd.to_datetime(temp.index)
            temp.index = index
            self.usdjpy1440_op = temp.iloc[:, 0]
            self.usdjpy1440_hi = temp.iloc[:, 1]
            self.usdjpy1440_lo = temp.iloc[:, 2]
            self.usdjpy1440_cl = temp.iloc[:, 3]
        # トレードの場合
        else:
            self.oanda = oandapy.API(
                environment=self.environment, access_token=self.access_token)
            # 1分足を格納する。
            self.audusd1_op = None
            self.audusd1_hi = None
            self.audusd1_lo = None
            self.audusd1_cl = None
            self.eurusd1_op = None
            self.eurusd1_hi = None
            self.eurusd1_lo = None
            self.eurusd1_cl = None
            self.gbpusd1_op = None
            self.gbpusd1_hi = None
            self.gbpusd1_lo = None
            self.gbpusd1_cl = None
            self.usdjpy1_op = None
            self.usdjpy1_hi = None
            self.usdjpy1_lo = None
            self.usdjpy1_cl = None
            # 5分足を格納する。
            self.audusd5_op = None
            self.audusd5_hi = None
            self.audusd5_lo = None
            self.audusd5_cl = None
            self.eurusd5_op = None
            self.eurusd5_hi = None
            self.eurusd5_lo = None
            self.eurusd5_cl = None
            self.gbpusd5_op = None
            self.gbpusd5_hi = None
            self.gbpusd5_lo = None
            self.gbpusd5_cl = None
            self.usdjpy5_op = None
            self.usdjpy5_hi = None
            self.usdjpy5_lo = None
            self.usdjpy5_cl = None
            # 15分足を格納する。
            self.audusd15_op = None
            self.audusd15_hi = None
            self.audusd15_lo = None
            self.audusd15_cl = None
            self.eurusd15_op = None
            self.eurusd15_hi = None
            self.eurusd15_lo = None
            self.eurusd15_cl = None
            self.gbpusd15_op = None
            self.gbpusd15_hi = None
            self.gbpusd15_lo = None
            self.gbpusd15_cl = None
            self.usdjpy15_op = None
            self.usdjpy15_hi = None
            self.usdjpy15_lo = None
            self.usdjpy15_cl = None
            # 30分足を格納する。
            self.audusd30_op = None
            self.audusd30_hi = None
            self.audusd30_lo = None
            self.audusd30_cl = None
            self.eurusd30_op = None
            self.eurusd30_hi = None
            self.eurusd30_lo = None
            self.eurusd30_cl = None
            self.gbpusd30_op = None
            self.gbpusd30_hi = None
            self.gbpusd30_lo = None
            self.gbpusd30_cl = None
            self.usdjpy30_op = None
            self.usdjpy30_hi = None
            self.usdjpy30_lo = None
            self.usdjpy30_cl = None
            # 1時間足を格納する。
            self.audusd60_op = None
            self.audusd60_hi = None
            self.audusd60_lo = None
            self.audusd60_cl = None
            self.eurusd60_op = None
            self.eurusd60_hi = None
            self.eurusd60_lo = None
            self.eurusd60_cl = None
            self.gbpusd60_op = None
            self.gbpusd60_hi = None
            self.gbpusd60_lo = None
            self.gbpusd60_cl = None
            self.usdjpy60_op = None
            self.usdjpy60_hi = None
            self.usdjpy60_lo = None
            self.usdjpy60_cl = None
            # 4時間足を格納する。
            self.audusd240_op = None
            self.audusd240_hi = None
            self.audusd240_lo = None
            self.audusd240_cl = None
            self.eurusd240_op = None
            self.eurusd240_hi = None
            self.eurusd240_lo = None
            self.eurusd240_cl = None
            self.gbpusd240_op = None
            self.gbpusd240_hi = None
            self.gbpusd240_lo = None
            self.gbpusd240_cl = None
            self.usdjpy240_op = None
            self.usdjpy240_hi = None
            self.usdjpy240_lo = None
            self.usdjpy240_cl = None
            # 日足を格納する。
            self.audusd1440_op = None
            self.audusd1440_hi = None
            self.audusd1440_lo = None
            self.audusd1440_cl = None
            self.eurusd1440_op = None
            self.eurusd1440_hi = None
            self.eurusd1440_lo = None
            self.eurusd1440_cl = None
            self.gbpusd1440_op = None
            self.gbpusd1440_hi = None
            self.gbpusd1440_lo = None
            self.gbpusd1440_cl = None
            self.usdjpy1440_op = None
            self.usdjpy1440_hi = None
            self.usdjpy1440_lo = None
            self.usdjpy1440_cl = None
        # 任意のSeriesで使用する。
        self.series1 = None
        self.series2 = None
        self.series3 = None
        self.series4 = None
        self.series5 = None
        self.series6 = None
        self.series7 = None
        self.series8 = None
        self.series9 = None
        self.series10 = None
 
    def ask(self, symbol):
        '''買値を得る。
          Args:
              symbol: 通貨ペア名。
          Returns:
              買値。
        '''
        if symbol == "AUDUSD":
            instrument = "AUD_USD"
        elif symbol == "EURUSD":
            instrument = "EUR_USD"
        elif symbol == "GBPUSD":
            instrument = "GBP_USD"
        else:  # symbol == "USDJPY"
            instrument = "USD_JPY"
        instruments = self.oanda.get_prices(instruments=instrument)
        ask = instruments["prices"][0]["ask"]
        return ask
 
    def backtest(self, strategy, parameter, symbol, timeframe, position,
    create_model, folder_model, remove_model, bounds, rranges, spread,
    optimization, de, min_trade, maxiter, popsize, seed, start_year,
    start_month, start_day, end_year, end_month, end_day):
        '''バックテストを行う。
          Args:
              strategy: 戦略関数。
              parameter: 最適化するパラメータ。
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              position: ポジションの設定。
              create_model: モデル作成関数。
              folder_model: モデルを保存するフォルダー。
              remove_model: モデル削除の設定。
              bounds: 差分進化で指定するパラメータの範囲。
              rranges: ブルートフォースで指定するパラメータの範囲。
              spread: スプレッド。
              optimization: 最適化の設定。
              de: 遺伝的アルゴリズムの設定。
              min_trade: 最低トレード数。
              maxiter: 最大繰り返し数。
              popsize: 世代数。
              seed: 乱数の種。
              start_year: 開始年。
              start_month: 開始月。
              start_day: 開始日。
              end_year: 終了年。
              end_month: 終了月。
              end_day: 終了日。
        '''
        start = datetime(start_year, start_month, start_day)
        end = datetime(end_year, end_month, end_day)
        model, array = create_model(
            self, symbol, timeframe, folder_model, remove_model, start, end)
        if optimization == 1:
            if de == 1:
                result = differential_evolution(
                    strategy, bounds, args=(self, symbol, timeframe, position,
                    model, array, spread, 1, min_trade, start, end),
                    maxiter=maxiter, popsize=popsize, seed=seed)
                parameter = result.x
            else:
                result = optimize.brute(
                    strategy, rranges, args=(self, symbol, timeframe, position,
                    model, array, spread, 1, min_trade, start, end),
                    finish=None)
                parameter = result
        report =  pd.DataFrame(
            index=[0], columns=["symbol", "start","end", "trades", "apr",
            "sharpe", "kelly", "parameter"])
        trades, apr, sharpe, kelly, ret = strategy(
            parameter, self, symbol, timeframe, position, model, array,
            spread=spread, optimization=0, min_trade=min_trade, start=start,
            end=end)
        report["symbol"] = symbol
        report["start"] = start.strftime("%Y-%m-%d")
        report["end"] = end.strftime("%Y-%m-%d")
        report["trades"] = trades
        report["apr"] = apr
        report["sharpe"] = sharpe
        report["kelly"] = kelly
        report["parameter"] = str(parameter)
        pd.set_option("line_width", 1000)
        print(report)
        cum_ret = ret.cumsum()
        graph = cum_ret.plot()
        graph.set_ylabel("cumulative return")
        plt.show()
 
    def bid(self, symbol):
        '''売値を得る。
          Args:
              symbol: 通貨ペア名。
          Returns:
              売値。
        '''
        if symbol == "AUDUSD":
            instrument = "AUD_USD"
        elif symbol == "EURUSD":
            instrument = "EUR_USD"
        elif symbol == "GBPUSD":
            instrument = "GBP_USD"
        else:  # symbol == "USDJPY"
            instrument = "USD_JPY"
        instruments = self.oanda.get_prices(instruments=instrument)
        bid = instruments["prices"][0]["bid"]
        return bid
 
    def calc_apr(self, ret, start, end):
        '''年率を計算する。
          Args:
              ret: リターン。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              年率。
        '''
        rate = (ret + 1.0).prod() - 1.0
        years = (end - start).total_seconds() / 60 / 60 / 24 / 365
        apr = rate / years
        return apr
 
    def calc_kelly(self, ret, start, end):
        '''ケリー基準による最適レバレッジを計算する。
          Args:
              ret: リターン。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              ケリー基準による最適レバレッジ。
        '''
        mean = ret.mean()
        std = ret.std()
        kelly = mean / (std * std)
        return kelly
 
    def calc_ret(self, symbol, timeframe, signal, spread, start, end):
        '''リターンを計算する。
          Args:
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              signal: シグナル。
              spread: スプレッド。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              年率。
        '''
        op = self.i_open(symbol, timeframe, 0, start=start, end=end)
        # 現在の足が買い、または売りで、1本前の足と状態が異なるとき、
        # コストが発生したと考える。
        cost = (((np.abs(signal)==1) & (signal!=signal.shift(1)))
            * spread)
        ret = ((op.shift(-1) - op) * signal - cost) / op
        ret = ret.fillna(0.0)
        ret[(ret==float("inf")) | (ret==float("-inf"))] = 0.0
        return ret
 
    def calc_sharpe(self, ret, start, end):
        '''シャープレシオを計算する。
          Args:
              ret: リターン。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              シャープレシオ。
        '''
        bars = len(ret)
        years = (end - start).total_seconds() / 60 / 60 / 24 / 365
        num_bar_per_year = bars / years
        mean = ret.mean()
        std = ret.std()
        sharpe = np.sqrt(num_bar_per_year) * mean / std
        return sharpe
 
    def calc_trades(self, signal, start, end):
        '''トレード数を計算する。
          Args:
              signal: シグナル。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              年率。
        '''
        trade = (np.abs(signal)==1) & (signal!=signal.shift(1))
        trades = trade[start:end].sum()
        return trades
 
    def i_above(self, symbol, timeframe, shift, start=None, end=None):
        '''現在の終値が直前の終値より上かどうかを返す（同値の場合は上と見なす）。
          Args:
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              shift: シフト。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              上か否か。
        '''
        # 終値を格納する（トレードの場合）
        if start is None:
            cl = self.i_close(symbol, timeframe, shift)
        # 終値を格納する（バックテストの場合）
        else:
            cl = self.i_close(
                symbol, timeframe, shift)[start-timedelta(minutes=timeframe)
                :end]
 
        # 現在の終値と直前の終値の差を格納する
        above = cl - cl.shift(1)
 
        # 差が0以上なら1、0未満なら0を格納する
        above[above>=0.0] = 1
        above[above<0.0] = 0
 
        # Nanはとりあえず上と見なして1で埋める
        above = above.fillna(0.0)
 
        # バックテストの場合は指定期間で切り取る
        if start is not None:
            above = above[start:end]
 
        return above
 
    def i_bandwalk(
    self, symbol, timeframe, period, shift, start=None, end=None):
        '''バンドウォークを返す。
          Args:
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              period: 期間。
              shift: シフト。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              バンドウォーク。
        '''
        if start is None:
            hi = self.i_high(symbol, timeframe, shift)
            lo = self.i_low(symbol, timeframe, shift)
            cl = self.i_close(symbol, timeframe, shift)
        else:
            hi = self.i_high(
                symbol, timeframe,
                shift)[start-timedelta(minutes=timeframe*period):end]
            lo = self.i_low(
                symbol, timeframe,
                shift)[start-timedelta(minutes=timeframe*period):end]
            cl = self.i_close(
                symbol, timeframe,
                shift)[start-timedelta(minutes=timeframe*period):end]
        mean = pd.rolling_mean(cl, period)
        up = (lo > mean) * 1
        up0 = up.copy()
        down = (hi < mean) * 1
        down0 = down.copy()
          # 処理に時間がかかるので処理回数を制限する。
        for i in range(int(period*1.5)):
            up = (up.shift(1) + 1) * up0
            down = (down.shift(1) + 1) * down0
        bandwalk = up - down
        if start == 0:
            bandwalk = bandwalk.fillna(0.0)
            return bandwalk
        else:
            bandwalk = bandwalk.fillna(0.0)[start:end]
            return bandwalk
 
    def i_chart(self, symbol, timeframe, period, shift, start=None, end=None):
        '''各時点から見た正規化された終値のチャートを返す。
          Args:
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              period: 期間。
              shift: シフト。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              各時点から見た正規化された終値のチャート。
        '''
        # 終値を格納する。
        if start is None:
            cl = self.i_close(symbol, timeframe, shift)
        else:
            cl = self.i_close(
                symbol, timeframe,
                shift)[start-timedelta(minutes=timeframe*period):end]
 
        # 各時点から見た正規化された終値のチャートを作成する。
        ma = pd.rolling_mean(cl, period)
        mstd = pd.rolling_std(cl, period)
        for i in range(period):
            normalize_cl = (cl.shift(i) - ma) / mstd
            if i == 0:
                chart = pd.DataFrame(
                    index=normalize_cl.index,
                    columns=["A", "B", "C", "D", "E", "F", "G", "H"])
                chart = chart.fillna(0)
                chart["A"] = (normalize_cl < -3.0) * 1
                chart["B"] = (((normalize_cl >= -3.0) & (normalize_cl < -2.0))
                    * 1)
                chart["C"] = (((normalize_cl >= -2.0) & (normalize_cl < -1.0))
                    * 1)
                chart["D"] = (((normalize_cl >= -1.0) & (normalize_cl < 0.0))
                    * 1)
                chart["E"] = ((normalize_cl >= 0.0) & (normalize_cl < 1.0)) * 1
                chart["F"] = ((normalize_cl >= 1.0) & (normalize_cl < 2.0)) * 1
                chart["G"] = ((normalize_cl >= 2.0) & (normalize_cl < 3.0)) * 1
                chart["H"] = (normalize_cl >= 3.0) * 1
            else:
                temp = pd.DataFrame(
                    index=normalize_cl.index,
                    columns=["A", "B", "C", "D", "E", "F", "G", "H"])
                temp = temp.fillna(0)
                temp["A"] = (normalize_cl < -3.0) * 1
                temp["B"] = (((normalize_cl >= -3.0) & (normalize_cl < -2.0))
                    * 1)
                temp["C"] = (((normalize_cl >= -2.0) & (normalize_cl < -1.0))
                    * 1)
                temp["D"] = ((normalize_cl >= -1.0) & (normalize_cl < 0.0)) * 1
                temp["E"] = ((normalize_cl >= 0.0) & (normalize_cl < 1.0)) * 1
                temp["F"] = ((normalize_cl >= 1.0) & (normalize_cl < 2.0)) * 1
                temp["G"] = ((normalize_cl >= 2.0) & (normalize_cl < 3.0)) * 1
                temp["H"] = (normalize_cl >= 3.0) * 1
                chart = pd.concat([chart, temp], axis=1)
                chart = chart.fillna(0.0)
                chart[(chart==float("inf")) | (chart==float("-inf"))] = 0.0
 
        # バックテストの場合、指定期間で切り取る
        if start is not None:
            chart = chart[start:end]
        return chart
 
    def i_close(self, symbol, timeframe, shift, start=None, end=None):
        '''終値を返す。
          Args:
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              shift: シフト。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              終値。
        '''
        if symbol == "AUDUSD":
            if timeframe == 1:
                cl = self.audusd1_cl.shift(shift)
            elif timeframe == 5:
                cl = self.audusd5_cl.shift(shift)
            elif timeframe == 15:
                cl = self.audusd15_cl.shift(shift)
            elif timeframe == 30:
                cl = self.audusd30_cl.shift(shift)
            elif timeframe == 60:
                cl = self.audusd60_cl.shift(shift)
            elif timeframe == 240:
                cl = self.audusd240_cl.shift(shift)
            else:  # timeframe == 1440
                cl = self.audusd1440_cl.shift(shift)
        elif symbol == "EURUSD":
            if timeframe == 1:
                cl = self.eurusd1_cl.shift(shift)
            elif timeframe == 5:
                cl = self.eurusd5_cl.shift(shift)
            elif timeframe == 15:
                cl = self.eurusd15_cl.shift(shift)
            elif timeframe == 30:
                cl = self.eurusd30_cl.shift(shift)
            elif timeframe == 60:
                cl = self.eurusd60_cl.shift(shift)
            elif timeframe == 240:
                cl = self.eurusd240_cl.shift(shift)
            else:  # timeframe == 1440
                cl = self.eurusd1440_cl.shift(shift)
        elif symbol == "GBPUSD":
            if timeframe == 1:
                cl = self.gbpusd1_cl.shift(shift)
            elif timeframe == 5:
                cl = self.gbpusd5_cl.shift(shift)
            elif timeframe == 15:
                cl = self.gbpusd15_cl.shift(shift)
            elif timeframe == 30:
                cl = self.gbpusd30_cl.shift(shift)
            elif timeframe == 60:
                cl = self.gbpusd60_cl.shift(shift)
            elif timeframe == 240:
                cl = self.gbpusd240_cl.shift(shift)
            else:  # timeframe == 1440
                cl = self.gbpusd1440_cl.shift(shift)
        else:  # symbol == "USDJPY"
            if timeframe == 1:
                cl = self.usdjpy1_cl.shift(shift)
            elif timeframe == 5:
                cl = self.usdjpy5_cl.shift(shift)
            elif timeframe == 15:
                cl = self.usdjpy15_cl.shift(shift)
            elif timeframe == 30:
                cl = self.usdjpy30_cl.shift(shift)
            elif timeframe == 60:
                cl = self.usdjpy60_cl.shift(shift)
            elif timeframe == 240:
                cl = self.usdjpy240_cl.shift(shift)
            else:  # timeframe == 1440
                cl = self.usdjpy1440_cl.shift(shift)
        if start is not None: cl = cl[start:end]
        return cl
 
    def i_high(self, symbol, timeframe, shift, start=None, end=None):
        '''高値を返す。
          Args:
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              shift: シフト。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              高値。
        '''
        if symbol == "AUDUSD":
            if timeframe == 1:
                hi = self.audusd1_hi.shift(shift)
            elif timeframe == 5:
                hi = self.audusd5_hi.shift(shift)
            elif timeframe == 15:
                hi = self.audusd15_hi.shift(shift)
            elif timeframe == 30:
                hi = self.audusd30_hi.shift(shift)
            elif timeframe == 60:
                hi = self.audusd60_hi.shift(shift)
            elif timeframe == 240:
                hi = self.audusd240_hi.shift(shift)
            else:  # timeframe == 1440
                hi = self.audusd1440_hi.shift(shift)
        elif symbol == "EURUSD":
            if timeframe == 1:
                hi = self.eurusd1_hi.shift(shift)
            elif timeframe == 5:
                hi = self.eurusd5_hi.shift(shift)
            elif timeframe == 15:
                hi = self.eurusd15_hi.shift(shift)
            elif timeframe == 30:
                hi = self.eurusd30_hi.shift(shift)
            elif timeframe == 60:
                hi = self.eurusd60_hi.shift(shift)
            elif timeframe == 240:
                hi = self.eurusd240_hi.shift(shift)
            else:  # timeframe == 1440
                hi = self.eurusd1440_hi.shift(shift)
        elif symbol == "GBPUSD":
            if timeframe == 1:
                hi = self.gbpusd1_hi.shift(shift)
            elif timeframe == 5:
                hi = self.gbpusd5_hi.shift(shift)
            elif timeframe == 15:
                hi = self.gbpusd15_hi.shift(shift)
            elif timeframe == 30:
                hi = self.gbpusd30_hi.shift(shift)
            elif timeframe == 60:
                hi = self.gbpusd60_hi.shift(shift)
            elif timeframe == 240:
                hi = self.gbpusd240_hi.shift(shift)
            else:  # timeframe == 1440
                hi = self.gbpusd1440_hi.shift(shift)
        else:  # symbol == "USDJPY"
            if timeframe == 1:
                hi = self.usdjpy1_hi.shift(shift)
            elif timeframe == 5:
                hi = self.usdjpy5_hi.shift(shift)
            elif timeframe == 15:
                hi = self.usdjpy15_hi.shift(shift)
            elif timeframe == 30:
                hi = self.usdjpy30_hi.shift(shift)
            elif timeframe == 60:
                hi = self.usdjpy60_hi.shift(shift)
            elif timeframe == 240:
                hi = self.usdjpy240_hi.shift(shift)
            else:  # timeframe == 1440
                hi = self.usdjpy1440_hi.shift(shift)
        if start is not None: hi = hi[start:end]
        return hi
 
    def i_low(self, symbol, timeframe, shift, start=None, end=None):
        '''安値を返す。
          Args:
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              shift: シフト。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              安値。
        '''
        if symbol == "AUDUSD":
            if timeframe == 1:
                lo = self.audusd1_lo.shift(shift)
            elif timeframe == 5:
                lo = self.audusd5_lo.shift(shift)
            elif timeframe == 15:
                lo = self.audusd15_lo.shift(shift)
            elif timeframe == 30:
                lo = self.audusd30_lo.shift(shift)
            elif timeframe == 60:
                lo = self.audusd60_lo.shift(shift)
            elif timeframe == 240:
                lo = self.audusd240_lo.shift(shift)
            else:  # timeframe == 1440
                lo = self.audusd1440_lo.shift(shift)
        elif symbol == "EURUSD":
            if timeframe == 1:
                lo = self.eurusd1_lo.shift(shift)
            elif timeframe == 5:
                lo = self.eurusd5_lo.shift(shift)
            elif timeframe == 15:
                lo = self.eurusd15_lo.shift(shift)
            elif timeframe == 30:
                lo = self.eurusd30_lo.shift(shift)
            elif timeframe == 60:
                lo = self.eurusd60_lo.shift(shift)
            elif timeframe == 240:
                lo = self.eurusd240_lo.shift(shift)
            else:  # timeframe == 1440
                lo = self.eurusd1440_lo.shift(shift)
        elif symbol == "GBPUSD":
            if timeframe == 1:
                lo = self.gbpusd1_lo.shift(shift)
            elif timeframe == 5:
                lo = self.gbpusd5_lo.shift(shift)
            elif timeframe == 15:
                lo = self.gbpusd15_lo.shift(shift)
            elif timeframe == 30:
                lo = self.gbpusd30_lo.shift(shift)
            elif timeframe == 60:
                lo = self.gbpusd60_lo.shift(shift)
            elif timeframe == 240:
                lo = self.gbpusd240_lo.shift(shift)
            else:  # timeframe == 1440
                lo = self.gbpusd1440_lo.shift(shift)
        else:  # symbol == "USDJPY"
            if timeframe == 1:
                lo = self.usdjpy1_lo.shift(shift)
            elif timeframe == 5:
                lo = self.usdjpy5_lo.shift(shift)
            elif timeframe == 15:
                lo = self.usdjpy15_lo.shift(shift)
            elif timeframe == 30:
                lo = self.usdjpy30_lo.shift(shift)
            elif timeframe == 60:
                lo = self.usdjpy60_lo.shift(shift)
            elif timeframe == 240:
                lo = self.usdjpy240_lo.shift(shift)
            else:  # timeframe == 1440
                lo = self.usdjpy1440_lo.shift(shift)
        if start is not None: lo = lo[start:end]
        return lo
 
    def i_lroc(self, symbol, timeframe, shift, start=None,
    end=None):
        '''対数変化率を返す。
          Args:
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              shift: シフト。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              対数変化率。
        '''
        if start is None:
            cl = self.i_close(symbol, timeframe, shift)
        else:
            cl = self.i_close(symbol, timeframe, shift)[start:end]
        cl_log = cl.apply(np.log)
        lroc = (cl_log - cl_log.shift(1)) * 100.0
        lroc = lroc.fillna(0.0)
        lroc[(lroc==float("inf")) | (lroc==float("-inf"))] = 0.0
        if start is not None: lroc = lroc[start:end]
        return lroc
 
    def i_open(self, symbol, timeframe, shift, start=None, end=None):
        '''始値を返す。
          Args:
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              shift: シフト。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              始値。
        '''
        if symbol == "AUDUSD":
            if timeframe == 1:
                op = self.audusd1_op.shift(shift)
            elif timeframe == 5:
                op = self.audusd5_op.shift(shift)
            elif timeframe == 15:
                op = self.audusd15_op.shift(shift)
            elif timeframe == 30:
                op = self.audusd30_op.shift(shift)
            elif timeframe == 60:
                op = self.audusd60_op.shift(shift)
            elif timeframe == 240:
                op = self.audusd240_op.shift(shift)
            else:  # timeframe == 1440
                op = self.audusd1440_op.shift(shift)
        elif symbol == "EURUSD":
            if timeframe == 1:
                op = self.eurusd1_op.shift(shift)
            elif timeframe == 5:
                op = self.eurusd5_op.shift(shift)
            elif timeframe == 15:
                op = self.eurusd15_op.shift(shift)
            elif timeframe == 30:
                op = self.eurusd30_op.shift(shift)
            elif timeframe == 60:
                op = self.eurusd60_op.shift(shift)
            elif timeframe == 240:
                op = self.eurusd240_op.shift(shift)
            else:  # timeframe == 1440
                op = self.eurusd1440_op.shift(shift)
        elif symbol == "GBPUSD":
            if timeframe == 1:
                op = self.gbpusd1_op.shift(shift)
            elif timeframe == 5:
                op = self.gbpusd5_op.shift(shift)
            elif timeframe == 15:
                op = self.gbpusd15_op.shift(shift)
            elif timeframe == 30:
                op = self.gbpusd30_op.shift(shift)
            elif timeframe == 60:
                op = self.gbpusd60_op.shift(shift)
            elif timeframe == 240:
                op = self.gbpusd240_op.shift(shift)
            else:  # timeframe == 1440
                op = self.gbpusd1440_op.shift(shift)
        else:  # symbol == "USDJPY"
            if timeframe == 1:
                op = self.usdjpy1_op.shift(shift)
            elif timeframe == 5:
                op = self.usdjpy5_op.shift(shift)
            elif timeframe == 15:
                op = self.usdjpy15_op.shift(shift)
            elif timeframe == 30:
                op = self.usdjpy30_op.shift(shift)
            elif timeframe == 60:
                op = self.usdjpy60_op.shift(shift)
            elif timeframe == 240:
                op = self.usdjpy240_op.shift(shift)
            else:  # timeframe == 1440
                op = self.usdjpy1440_op.shift(shift)
        if start is not None: op = op[start:end]
        return op
 
    def i_roc(self, symbol, timeframe, period, shift, start=None, end=None):
        '''ROCを返す。
          Args:
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              period: 期間。
              shift: シフト。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              ROC。
        '''
        if start is None:
            cl = self.i_close(symbol, timeframe, shift)
        else:
            cl = self.i_close(
                symbol, timeframe,
                shift)[start-timedelta(minutes=timeframe*period):end]
        roc = (cl - cl.shift(period)) / cl.shift(period) * 100.0
        roc = roc.fillna(0.0)
        roc[(roc==float("inf")) | (roc==float("-inf"))] = 0.0
        if start is not None: roc = roc[start:end]
        return roc
 
    def i_rocs(self, symbol, timeframe, period, shift, start=None,
    end=None):
        '''各時点から見たROCを返す。
          Args:
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              period: 期間。
              shift: シフト。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              各時点から見たROC。
        '''
        # 終値を格納する。
        if start is None:
            cl = self.i_close(symbol, timeframe, shift)
        else:
            cl = self.i_close(
                symbol, timeframe,
                shift)[start-timedelta(minutes=timeframe*period):end]
 
        # 各時点から見たROCを格納する。
        for i in range(period):
            if i == 0:
                rocs = (cl - cl.shift(1)) / cl.shift(1) * 100.0
            else:
                rocs = pd.concat(
                    [rocs, (cl - cl.shift(1+i)) / cl.shift(1+i) * 100.0],
                    axis=1)
        rocs = rocs.fillna(0)
        rocs[(rocs==float("inf")) | (rocs==float("-inf"))] = 0.0
 
        # バックテストの場合、指定期間で切り取る
        if start is not None: rocs = rocs[start:end]
 
        return rocs
 
    def i_z_score(self, symbol, timeframe, period, shift, start=None,
    end=None):
        '''zスコアを返す。
          Args:
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              period: 期間。
              shift: シフト。
              start: 開始年月日。
              end: 終了年月日。
          Returns:
              zスコア。
        '''
        if start is None:
            cl = self.i_close(symbol, timeframe, shift)
        else:
            cl = self.i_close(
                symbol, timeframe,
                shift)[start-timedelta(minutes=timeframe*period):end]
        ma = pd.rolling_mean(cl, period)
        mstd = pd.rolling_std(cl, period)
        z_score = (cl - ma) / mstd
        z_score = z_score.fillna(0.0)
        z_score[(z_score==float("inf")) | (z_score==float("-inf"))] = 0.0
        if start is not None: z_score = z_score[start:end]
        return z_score
 
    def order_close(self, ticket):
        '''決済注文を送信する。
          Args:
              ticket: チケット番号。
        '''
        self.oanda.close_trade(self.account_id, ticket)
 
    def order_send(self, symbol, lots, side):
        '''新規注文を送信する。
          Args:
              symbol: 通貨ペア名。
              lots: ロット数。
              side: 売買の種別
          Returns:
              チケット番号。
        '''
        if symbol == "AUDUSD":
            instrument = "AUD_USD"
        elif symbol == "EURUSD":
            instrument = "EUR_USD"
        elif symbol == "GBPUSD":
            instrument = "GBP_USD"
        else:  # symbol == "USDJPY"
            instrument = "USD_JPY"
        response = self.oanda.create_order(account_id=self.account_id,
            instrument=instrument, units=int(lots*10000), side=side,
            type="market")
        ticket = response["tradeOpened"]["id"]
        return ticket
 
    def time_day_of_week(self, index):
        '''曜日を返す。
          Args:
              index: インデックス。
          Returns:
              曜日。
        '''
        time_day_of_week = pd.Series(index.dayofweek, index=index)
        return time_day_of_week
 
    def time_hour(self, index):
        '''時を返す。
          Args:
              index: インデックス。
          Returns:
              時。
        '''
        time_hour = pd.Series(index.hour, index=index)
        return time_hour
 
    def time_minute(self, index):
        '''分を返す。
          Args:
              index: インデックス。
          Returns:
              分。
        '''
        time_minute = pd.Series(index.minute, index=index)
        return time_minute
 
    def trade(self, strategy, parameter, symbol, timeframe, position, model,
    array, spread, lots, ea, folder_ea, file_ea, ticket, pos, event):
        '''トレードを行う。
          Args:
              strategy: 戦略関数。
              parameter: デフォルトのパラメータ。
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              position: ポジションの設定。
              model: モデル。
              array: 各種数値を格納した配列。
              spread: スプレッド。
              lots: ロット数。
              ea: EAの設定。
              folder_ea: EAがアクセスするフォルダー名。
              file_ea: EAがアクセスするファイル名。
              ticket: チケット番号。
              pos: ポジション。
              event: イベント。
        '''
        while True:
            # エラー処理をする。
            try:
                signal = strategy(
                    parameter, self, symbol, timeframe, position, model, array)
                end_row = len(signal) - 1
                # 新規注文を送信する
                ask = self.ask(symbol)  # 買値を得る
                bid = self.bid(symbol)  # 買値を得る
                if (pos == 0 and ask - bid <= spread):
                    if (signal[end_row] == 1):
                        ticket = self.order_send(symbol, lots, "buy")
                        pos = 1
                    elif (signal[end_row] == -1):
                        ticket = self.order_send(symbol, lots, "sell")
                        pos = -1
                # 決済注文を送信する
                else:
                    if pos == 1 and signal[end_row] != 1:
                        self.order_close(ticket)
                        ticket = 0
                        pos = 0
                    elif pos == -1 and signal[end_row] != -1:
                        self.order_close(ticket)
                        ticket = 0
                        pos = 0
                # EAにシグナルを送信する場合
                if ea == 1:
                    filename = folder_ea + "/" + file_ea
                    f = open(filename, "w")
                    # シグナルに2を加えて0をなくす。
                    f.write(str(int(signal[end_row] + 2)))
                    f.close()
                # ポジション情報を出力
                now = datetime.now()
                account = self.oanda.get_account(self.account_id)
                positions = self.oanda.get_positions(self.account_id)
                if int(account["openTrades"]) != 0:
                    side = positions["positions"][0]["side"]
                    avg_price = round(positions["positions"][0]["avgPrice"], 3)
                    if side == "buy":
                        pnl = (bid - avg_price) * lots * 10000
                        price = bid
                    else:
                        pnl = (avg_price - ask) * lots * 10000
                        price = ask
                    pnl = round(pnl, 3)
                    price = round(price, 3)
                    print(
                        now.strftime("%Y-%m-%d %H:%M:%S"), symbol, lots, side,
                        avg_price, price, pnl)
                else:
                    print(
                        now.strftime("%Y-%m-%d %H:%M:%S"), "None", 0, "None",
                        0.0, 0.0, 0.0)
                event.set()
                event.clear()
            except:
                print("trade()関数でエラーが発生しましたので終了します。")
                break
 
    def update_data(self, symbol, timeframe, count, event):
        '''データをアップデートする。
          Args:
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              count: 取得するバー数。
              event: イベント。
        '''
        if symbol == "AUDUSD":
            instrument = "AUD_USD"
        elif symbol == "EURUSD":
            instrument = "EUR_USD"
        elif symbol == "GBPUSD":
            instrument = "GBP_USD"
        else:  # symbol == "USDJPY"
            instrument = "USD_JPY"
        if timeframe == 1:
            granularity = "M1"
        elif timeframe == 5:
            granularity = "M5"
        elif timeframe == 15:
            granularity = "M15"
        elif timeframe == 30:
            granularity = "M30"
        elif timeframe == 60:
            granularity = "H1"
        elif timeframe == 240:
            granularity = "H4"
        else:  # timeframe == 1440
            granularity = "D"
        while True:
            event.wait()
            temp = self.oanda.get_history(
                instrument=instrument, granularity=granularity, count=count)
            self.time = pd.to_datetime(temp["candles"][count-1]["time"])
            index = pd.Series(np.zeros(count))
            op = pd.Series(np.zeros(count))
            hi = pd.Series(np.zeros(count))
            lo = pd.Series(np.zeros(count))
            cl = pd.Series(np.zeros(count))
            for i in range(count):
                index[i] = temp["candles"][i]["time"]
                op.iloc[i] = temp["candles"][i]["openBid"]
                hi.iloc[i] = temp["candles"][i]["highBid"]
                lo.iloc[i] = temp["candles"][i]["lowBid"]
                cl.iloc[i] = temp["candles"][i]["closeBid"]
            # NYクロージングの時間に合わせる
            index = pd.to_datetime(index) + timedelta(hours=2)
            # AUDUSD
            if symbol == "AUDUSD":
                if timeframe == 1:
                    self.audusd1_op = op
                    self.audusd1_hi = hi
                    self.audusd1_lo = lo
                    self.audusd1_cl = cl
                    self.audusd1_op.index = index
                    self.audusd1_hi.index = index
                    self.audusd1_lo.index = index
                    self.audusd1_cl.index = index
                elif timeframe == 5:
                    self.audusd5_op = op
                    self.audusd5_hi = hi
                    self.audusd5_lo = lo
                    self.audusd5_cl = cl
                    self.audusd5_op.index = index
                    self.audusd5_hi.index = index
                    self.audusd5_lo.index = index
                    self.audusd5_cl.index = index
                elif timeframe == 15:
                    self.audusd15_op = op
                    self.audusd15_hi = hi
                    self.audusd15_lo = lo
                    self.audusd15_cl = cl
                    self.audusd15_op.index = index
                    self.audusd15_hi.index = index
                    self.audusd15_lo.index = index
                    self.audusd15_cl.index = index
                elif timeframe == 30:
                    self.audusd30_op = op
                    self.audusd30_hi = hi
                    self.audusd30_lo = lo
                    self.audusd30_cl = cl
                    self.audusd30_op.index = index
                    self.audusd30_hi.index = index
                    self.audusd30_lo.index = index
                    self.audusd30_cl.index = index
                elif timeframe == 60:
                    self.audusd60_op = op
                    self.audusd60_hi = hi
                    self.audusd60_lo = lo
                    self.audusd60_cl = cl
                    self.audusd60_op.index = index
                    self.audusd60_hi.index = index
                    self.audusd60_lo.index = index
                    self.audusd60_cl.index = index
                elif timeframe == 240:
                    self.audusd240_op = op
                    self.audusd240_hi = hi
                    self.audusd240_lo = lo
                    self.audusd240_cl = cl
                    self.audusd240_op.index = index
                    self.audusd240_hi.index = index
                    self.audusd240_lo.index = index
                    self.audusd240_cl.index = index
                else:  # timeframe == 1440
                    self.audusd1440_op = op
                    self.audusd1440_hi = hi
                    self.audusd1440_lo = lo
                    self.audusd1440_cl = cl
                    self.audusd1440_op.index = index
                    self.audusd1440_hi.index = index
                    self.audusd1440_lo.index = index
                    self.audusd1440_cl.index = index
            # EURUSD
            elif symbol == "EURUSD":
                if timeframe == 1:
                    self.eurusd1_op = op
                    self.eurusd1_hi = hi
                    self.eurusd1_lo = lo
                    self.eurusd1_cl = cl
                    self.eurusd1_op.index = index
                    self.eurusd1_hi.index = index
                    self.eurusd1_lo.index = index
                    self.eurusd1_cl.index = index
                elif timeframe == 5:
                    self.eurusd5_op = op
                    self.eurusd5_hi = hi
                    self.eurusd5_lo = lo
                    self.eurusd5_cl = cl
                    self.eurusd5_op.index = index
                    self.eurusd5_hi.index = index
                    self.eurusd5_lo.index = index
                    self.eurusd5_cl.index = index
                elif timeframe == 15:
                    self.eurusd15_op = op
                    self.eurusd15_hi = hi
                    self.eurusd15_lo = lo
                    self.eurusd15_cl = cl
                    self.eurusd15_op.index = index
                    self.eurusd15_hi.index = index
                    self.eurusd15_lo.index = index
                    self.eurusd15_cl.index = index
                elif timeframe == 30:
                    self.eurusd30_op = op
                    self.eurusd30_hi = hi
                    self.eurusd30_lo = lo
                    self.eurusd30_cl = cl
                    self.eurusd30_op.index = index
                    self.eurusd30_hi.index = index
                    self.eurusd30_lo.index = index
                    self.eurusd30_cl.index = index
                elif timeframe == 60:
                    self.eurusd60_op = op
                    self.eurusd60_hi = hi
                    self.eurusd60_lo = lo
                    self.eurusd60_cl = cl
                    self.eurusd60_op.index = index
                    self.eurusd60_hi.index = index
                    self.eurusd60_lo.index = index
                    self.eurusd60_cl.index = index
                elif timeframe == 240:
                    self.eurusd240_op = op
                    self.eurusd240_hi = hi
                    self.eurusd240_lo = lo
                    self.eurusd240_cl = cl
                    self.eurusd240_op.index = index
                    self.eurusd240_hi.index = index
                    self.eurusd240_lo.index = index
                    self.eurusd240_cl.index = index
                else:  # timeframe == 1440
                    self.eurusd1440_op = op
                    self.eurusd1440_hi = hi
                    self.eurusd1440_lo = lo
                    self.eurusd1440_cl = cl
                    self.eurusd1440_op.index = index
                    self.eurusd1440_hi.index = index
                    self.eurusd1440_lo.index = index
                    self.eurusd1440_cl.index = index
            # GBPUSD
            elif symbol == "GBPUSD":
                if timeframe == 1:
                    self.gbpusd1_op = op
                    self.gbpusd1_hi = hi
                    self.gbpusd1_lo = lo
                    self.gbpusd1_cl = cl
                    self.gbpusd1_op.index = index
                    self.gbpusd1_hi.index = index
                    self.gbpusd1_lo.index = index
                    self.gbpusd1_cl.index = index
                elif timeframe == 5:
                    self.gbpusd5_op = op
                    self.gbpusd5_hi = hi
                    self.gbpusd5_lo = lo
                    self.gbpusd5_cl = cl
                    self.gbpusd5_op.index = index
                    self.gbpusd5_hi.index = index
                    self.gbpusd5_lo.index = index
                    self.gbpusd5_cl.index = index
                elif timeframe == 15:
                    self.gbpusd15_op = op
                    self.gbpusd15_hi = hi
                    self.gbpusd15_lo = lo
                    self.gbpusd15_cl = cl
                    self.gbpusd15_op.index = index
                    self.gbpusd15_hi.index = index
                    self.gbpusd15_lo.index = index
                    self.gbpusd15_cl.index = index
                elif timeframe == 30:
                    self.gbpusd30_op = op
                    self.gbpusd30_hi = hi
                    self.gbpusd30_lo = lo
                    self.gbpusd30_cl = cl
                    self.gbpusd30_op.index = index
                    self.gbpusd30_hi.index = index
                    self.gbpusd30_lo.index = index
                    self.gbpusd30_cl.index = index
                elif timeframe == 60:
                    self.gbpusd60_op = op
                    self.gbpusd60_hi = hi
                    self.gbpusd60_lo = lo
                    self.gbpusd60_cl = cl
                    self.gbpusd60_op.index = index
                    self.gbpusd60_hi.index = index
                    self.gbpusd60_lo.index = index
                    self.gbpusd60_cl.index = index
                elif timeframe == 240:
                    self.gbpusd240_op = op
                    self.gbpusd240_hi = hi
                    self.gbpusd240_lo = lo
                    self.gbpusd240_cl = cl
                    self.gbpusd240_op.index = index
                    self.gbpusd240_hi.index = index
                    self.gbpusd240_lo.index = index
                    self.gbpusd240_cl.index = index
                else:  # timeframe == 1440
                    self.gbpusd1440_op = op
                    self.gbpusd1440_hi = hi
                    self.gbpusd1440_lo = lo
                    self.gbpusd1440_cl = cl
                    self.gbpusd1440_op.index = index
                    self.gbpusd1440_hi.index = index
                    self.gbpusd1440_lo.index = index
                    self.gbpusd1440_cl.index = index
            # USDJPY
            else:  # symbol == "USDJPY"
                if timeframe == 1:
                    self.usdjpy1_op = op
                    self.usdjpy1_hi = hi
                    self.usdjpy1_lo = lo
                    self.usdjpy1_cl = cl
                    self.usdjpy1_op.index = index
                    self.usdjpy1_hi.index = index
                    self.usdjpy1_lo.index = index
                    self.usdjpy1_cl.index = index
                elif timeframe == 5:
                    self.usdjpy5_op = op
                    self.usdjpy5_hi = hi
                    self.usdjpy5_lo = lo
                    self.usdjpy5_cl = cl
                    self.usdjpy5_op.index = index
                    self.usdjpy5_hi.index = index
                    self.usdjpy5_lo.index = index
                    self.usdjpy5_cl.index = index
                elif timeframe == 15:
                    self.usdjpy15_op = op
                    self.usdjpy15_hi = hi
                    self.usdjpy15_lo = lo
                    self.usdjpy15_cl = cl
                    self.usdjpy15_op.index = index
                    self.usdjpy15_hi.index = index
                    self.usdjpy15_lo.index = index
                    self.usdjpy15_cl.index = index
                elif timeframe == 30:
                    self.usdjpy30_op = op
                    self.usdjpy30_hi = hi
                    self.usdjpy30_lo = lo
                    self.usdjpy30_cl = cl
                    self.usdjpy30_op.index = index
                    self.usdjpy30_hi.index = index
                    self.usdjpy30_lo.index = index
                    self.usdjpy30_cl.index = index
                elif timeframe == 60:
                    self.usdjpy60_op = op
                    self.usdjpy60_hi = hi
                    self.usdjpy60_lo = lo
                    self.usdjpy60_cl = cl
                    self.usdjpy60_op.index = index
                    self.usdjpy60_hi.index = index
                    self.usdjpy60_lo.index = index
                    self.usdjpy60_cl.index = index
                elif timeframe == 240:
                    self.usdjpy240_op = op
                    self.usdjpy240_hi = hi
                    self.usdjpy240_lo = lo
                    self.usdjpy240_cl = cl
                    self.usdjpy240_op.index = index
                    self.usdjpy240_hi.index = index
                    self.usdjpy240_lo.index = index
                    self.usdjpy240_cl.index = index
                else:  # timeframe == 1440
                    self.usdjpy1440_op = op
                    self.usdjpy1440_hi = hi
                    self.usdjpy1440_lo = lo
                    self.usdjpy1440_cl = cl
                    self.usdjpy1440_op.index = index
                    self.usdjpy1440_hi.index = index
                    self.usdjpy1440_lo.index = index
                    self.usdjpy1440_cl.index = index
 
    def walk_forward_test(self, strategy, parameter, symbol, timeframe,
    position, create_model, folder_model, remove_model, bounds, rranges,
    spread, optimization, de, min_trade, maxiter, popsize, seed,
    in_sample_period, out_of_sample_period, start_year, start_month, start_day,
    end_year, end_month, end_day):
        '''ウォークフォワードテストを行う。
          Args:
              strategy: 戦略関数。
              parameter: 最適化するパラメータ。
              symbol: 通貨ペア名。
              timeframe: タイムフレーム。
              position: ポジションの設定。
              create_model: モデル作成関数。
              folder_model: モデルを保存するフォルダー。
              remove_model: モデル削除の設定。
              bounds: 差分進化で指定するパラメータの範囲。
              rranges: ブルートフォースで指定するパラメータの範囲。
              spread: スプレッド。
              optimization: 最適化の設定。
              de: 遺伝的アルゴリズムの設定。
              min_trade: 最低トレード数。
              maxiter: 最大繰り返し数。
              popsize: 世代数。
              seed: 乱数の種。
              in_sample_period: インサンプル期間
              out_of_sample_period: アウトオブサンプル期間
              start_year: 開始年。
              start_month: 開始月。
              start_day: 開始日。
              end_year: 終了年。
              end_month: 終了月。
              end_day: 終了日。
        '''
        end_test = datetime(start_year, start_month, start_day)
        report =  pd.DataFrame(
            index=range(1000),
            columns=["symbol", "start_train","end_train", "start_test",
            "end_test", "trades", "apr", "sharpe", "kelly", "parameter"])
        i = 0
        while True:    
            start_train = (datetime(start_year, start_month, start_day)
                + timedelta(days=out_of_sample_period*i))
            end_train = (start_train + timedelta(days=in_sample_period)
                - timedelta(minutes=timeframe))
            start_test = end_train + timedelta(minutes=timeframe)
            end_test = (start_test + timedelta(days=out_of_sample_period)
                - timedelta(minutes=timeframe))
            model, array = create_model(
                self, symbol, timeframe, folder_model, remove_model,
                start_train, end_train)
            if end_test > datetime(end_year, end_month, end_day):
                break
            if de == 1:
                result = differential_evolution(
                    strategy, bounds, args=(self, symbol, timeframe, position, 
                    model, array, spread, 1, min_trade, start_train, end_train),
                    maxiter=maxiter, popsize=popsize, seed=seed)
                parameter = result.x
            else:
                result = optimize.brute(
                    strategy, rranges, args=(self, symbol, timeframe, position, 
                    model, array, spread, 1, min_trade, start_train, end_train),
                    finish=None)
                parameter = result
            trades, apr, sharpe, kelly, ret = strategy(
                parameter, self, symbol, timeframe, position, model, array,
                spread=spread, optimization=0, min_trade=min_trade,
                start=start_test, end=end_test)
            if i == 0:
                ret_all = ret
            else:
                ret_all = ret_all.append(ret)
            report.iloc[i][0] = symbol
            report.iloc[i][1] = start_train.strftime("%Y-%m-%d")
            report.iloc[i][2] = end_train.strftime("%Y-%m-%d")
            report.iloc[i][3] = start_test.strftime("%Y-%m-%d")
            report.iloc[i][4] = end_test.strftime("%Y-%m-%d")
            report.iloc[i][5] = trades
            report.iloc[i][6] = apr
            report.iloc[i][7] = sharpe
            report.iloc[i][8] = kelly
            report.iloc[i][9] = str(parameter)
            i = i + 1
        trades = report.iloc[0:i, 5]
        apr = report.iloc[0:i, 6]
        sharpe = report.iloc[0:i, 7]
        kelly = report.iloc[0:i, 8]
        report.iloc[i][0] = symbol
        report.iloc[i][1] = ""
        report.iloc[i][2] = ""
        report.iloc[i][3] = report.iloc[0][3]
        report.iloc[i][4] = report.iloc[i-1][4]
        report.iloc[i][5] = int(trades.sum() / i)
        report.iloc[i][6] = apr.sum() / i
        report.iloc[i][7] = sharpe.sum() / i
        report.iloc[i][8] = kelly.sum() / i
        report.iloc[i][9] = ""
        pd.set_option("line_width", 1000)
        print(report.iloc[:i+1, ])
        cum_ret = ret_all.cumsum()
        graph = cum_ret.plot()
        graph.set_ylabel("cumulative return")
        plt.show()

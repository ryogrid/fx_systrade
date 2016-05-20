# coding: utf-8
  
import forex_system
import numpy as np
import os
import pandas as pd
import sys
import threading
import time
from datetime import datetime
from sklearn import linear_model
from sklearn.externals import joblib
  
# 一般設定
SYMBOL = "USDJPY"
TIMEFRAME = 5
SPREAD = 0.004
THRESHOLD = 2.0
POSITION = 2  # 0: 買いのみ 1: 売りのみ 2: 売買両方
# バックテスト設定
START_YEAR = 2006
START_MONTH = 1
START_DAY = 1
END_YEAR = 2015
END_MONTH = 12
END_DAY = 31
# モデル設定
FOLDER_MODEL = "py/linear_regression"
REMOVE_MODEL = 1  # 0: モデルを削除しない 1: モデルを削除する
# 最適化設定
WFT = 1  # 0: バックテスト 1: ウォークフォワードテスト
OPTIMIZATION = 1  # 0: 最適化なし 1: 最適化あり
DE = 0  # 0: ブルートフォース 1: 差分進化
MIN_TRADE = 260
START_THRESHOLD = 0.5
END_THRESHOLD = 5.0
STEP_THRESHOLD = 0.5
MAXITER = 10
POPSIZE = 3
SEED = 1
# ウォークフォワードテスト設定
IN_SAMPLE_PERIOD = 365
OUT_OF_SAMPLE_PERIOD = 91
# トレード設定
LOTS = 0.1  # 1ロット=1万通貨単位
# EA設定
EA = 0  # 0: EAにシグナル送信なし 1: EAにシグナル送信あり
FOLDER_EA = ("./.wine/drive_c/Program Files (x86)/FXDD Malta - MetaTrader 4/"
    + "MQL4/Files")
FILE_EA = "linear_regression.csv"  # ファイル名は31文字以内にする。
  
def strategy(parameter, fs, symbol, timeframe, position, model, array,
spread=0.0, optimization=0, min_trade=0, start=None, end=None):
    '''戦略を記述する。
      Args:
          parameter: 最適化したパラメータ。
          fs: ForexSystemクラスのインスタンス。
          symbol: 通貨ペア名。
          timeframe: タイムフレーム。
          position: ポジションの設定。
          model: モデル。
          array: 各種数値を格納した配列。
          spread: スプレッド。
          optimization: 最適化の設定。
          min_trade: 最低トレード数。
          start: 開始年月日。
          end: 終了年月日。
      Returns:
          トレードの場合はシグナル、バックテストの場合はパフォーマンス。
    '''
    threshold = float(parameter)
    (pred_mean, pred_std, bandwalk5_1_mean, bandwalk5_1_std, bandwalk10_1_mean,
        bandwalk10_1_std, bandwalk15_1_mean, bandwalk15_1_std,
        bandwalk20_1_mean, bandwalk20_1_std, bandwalk25_1_mean,
        bandwalk25_1_std, bandwalk30_1_mean, bandwalk30_1_std,
        bandwalk35_1_mean, bandwalk35_1_std, bandwalk40_1_mean,
        bandwalk40_1_std, bandwalk45_1_mean, bandwalk45_1_std,
        bandwalk50_1_mean, bandwalk50_1_std) = array
    # 説明変数を格納する。
    if start is None:
        bandwalk5_1 = fs.i_bandwalk(symbol, timeframe, 5, 1)
        bandwalk10_1 = fs.i_bandwalk(symbol, timeframe, 10, 1)
        bandwalk15_1 = fs.i_bandwalk(symbol, timeframe, 15, 1)
        bandwalk20_1 = fs.i_bandwalk(symbol, timeframe, 20, 1)
        bandwalk25_1 = fs.i_bandwalk(symbol, timeframe, 25, 1)
        bandwalk30_1 = fs.i_bandwalk(symbol, timeframe, 30, 1)
        bandwalk35_1 = fs.i_bandwalk(symbol, timeframe, 35, 1)
        bandwalk40_1 = fs.i_bandwalk(symbol, timeframe, 40, 1)
        bandwalk45_1 = fs.i_bandwalk(symbol, timeframe, 45, 1)
        bandwalk50_1 = fs.i_bandwalk(symbol, timeframe, 50, 1)
    else:
        bandwalk5_1 = fs.i_bandwalk(
            symbol, timeframe, 5, 1, start=start, end=end)
        bandwalk10_1 = fs.i_bandwalk(
            symbol, timeframe, 10, 1, start=start, end=end)
        bandwalk15_1 = fs.i_bandwalk(
            symbol, timeframe, 15, 1, start=start, end=end)
        bandwalk20_1 = fs.i_bandwalk(
            symbol, timeframe, 20, 1, start=start, end=end)
        bandwalk25_1 = fs.i_bandwalk(
            symbol, timeframe, 25, 1, start=start, end=end)
        bandwalk30_1 = fs.i_bandwalk(
            symbol, timeframe, 30, 1, start=start, end=end)
        bandwalk35_1 = fs.i_bandwalk(
            symbol, timeframe, 35, 1, start=start, end=end)
        bandwalk40_1 = fs.i_bandwalk(
            symbol, timeframe, 40, 1, start=start, end=end)
        bandwalk45_1 = fs.i_bandwalk(
            symbol, timeframe, 45, 1, start=start, end=end)
        bandwalk50_1 = fs.i_bandwalk(
            symbol, timeframe, 50, 1, start=start, end=end)
    # 説明変数を正規化する。
    bandwalk5_1 = (bandwalk5_1 - bandwalk5_1_mean) / bandwalk5_1_std
    bandwalk10_1 = (bandwalk10_1 - bandwalk10_1_mean) / bandwalk10_1_std
    bandwalk15_1 = (bandwalk15_1 - bandwalk15_1_mean) / bandwalk15_1_std
    bandwalk20_1 = (bandwalk20_1 - bandwalk20_1_mean) / bandwalk20_1_std
    bandwalk25_1 = (bandwalk25_1 - bandwalk25_1_mean) / bandwalk25_1_std
    bandwalk30_1 = (bandwalk30_1 - bandwalk30_1_mean) / bandwalk30_1_std
    bandwalk35_1 = (bandwalk35_1 - bandwalk35_1_mean) / bandwalk35_1_std
    bandwalk40_1 = (bandwalk40_1 - bandwalk40_1_mean) / bandwalk40_1_std
    bandwalk45_1 = (bandwalk45_1 - bandwalk45_1_mean) / bandwalk45_1_std
    bandwalk50_1 = (bandwalk50_1 - bandwalk50_1_mean) / bandwalk50_1_std
    x = pd.concat(
        [bandwalk5_1, bandwalk10_1, bandwalk15_1, bandwalk20_1, bandwalk25_1,
        bandwalk30_1, bandwalk35_1, bandwalk40_1, bandwalk45_1, bandwalk50_1],
        axis=1)
    x = x.fillna(0.0)  # とりあえず0で埋める。
    x[(x==float("inf")) | (x==float("-inf"))] = 0.0  # とりあえず0で埋める。
    # 予測値を計算する。
    pred = model.predict(x)
    pred = pd.Series(pred, index=x.index)
    # シグナルを計算する。
    #pred_mean = 0.0  # デフォルト値を設定したい場合はコメントアウトを外す。
    #pred_std = 1.0  # デフォルト値を設定したい場合はコメントアウトを外す。
    longs_entry = ((pred - pred_mean) / pred_std >= threshold) * 1
    longs_exit = ((pred - pred_mean) / pred_std <= 0.0) * 1
    shorts_entry = ((pred - pred_mean) / pred_std <= -threshold) * 1
    shorts_exit = ((pred - pred_mean) / pred_std >= 0.0) * 1
    longs = longs_entry.copy()
    longs[longs==0] = np.nan
    longs[longs_exit==1] = 0
    longs = longs.fillna(method="ffill")
    shorts = -shorts_entry.copy()
    shorts[shorts==0] = np.nan
    shorts[shorts_exit==1] = 0
    shorts = shorts.fillna(method="ffill")
    if position == 0:
        signal = longs
    elif position == 1:
        signal = shorts
    else:  # position == 2
        signal = longs + shorts
    signal = signal.fillna(0)
    # トレードである場合はシグナルを返して終了する。
    if start is None: return signal
    # パフォーマンスを計算する。
    ret = fs.calc_ret(symbol, timeframe, signal, spread, start=start, end=end)
    trades = fs.calc_trades(signal, start=start, end=end)
    apr = fs.calc_apr(ret, start=start, end=end)
    sharpe = fs.calc_sharpe(ret, start=start, end=end)
    kelly = fs.calc_kelly(ret, start=start, end=end)
    years = (end - start).total_seconds() / 60 / 60 / 24 / 365
    # 1年当たりのトレード数が最低トレード数に満たない場合、
    # 年率、シャープレシオを0にする。
    if trades / years < min_trade:
        apr = 0.0
        sharpe = 0.0
        kelly = 0.0
    # 最適化しない場合、各パフォーマンスを返す。
    if optimization == 0:
        return trades, apr, sharpe, kelly, ret
    # 最適化する場合、シャープレシオのみ符号を逆にして返す
    # （最適化関数が最小値のみ求めるため）。
    else:
        return -sharpe
  
def create_model(fs, symbol, timeframe, folder_model, remove_model, start, end):
    '''モデルを記述する。
      Args:
          fs: ForexSystemクラスのインスタンス。
          symbol: 通貨ペア名。
          timeframe: タイムフレーム。
          folder_model: モデルを保存するフォルダー。
          remove_model: モデル削除の設定。
          start: 開始年月日。
          end: 終了年月日。
      Returns:
          モデル、各種数値を格納した配列。
    '''
    # 説明変数を格納する。
    bandwalk5_1 = fs.i_bandwalk(symbol, timeframe, 5, 1, start=start, end=end)
    bandwalk10_1 = fs.i_bandwalk(symbol, timeframe, 10, 1, start=start, end=end)
    bandwalk15_1 = fs.i_bandwalk(symbol, timeframe, 15, 1, start=start, end=end)
    bandwalk20_1 = fs.i_bandwalk(symbol, timeframe, 20, 1, start=start, end=end)
    bandwalk25_1 = fs.i_bandwalk(symbol, timeframe, 25, 1, start=start, end=end)
    bandwalk30_1 = fs.i_bandwalk(symbol, timeframe, 30, 1, start=start, end=end)
    bandwalk35_1 = fs.i_bandwalk(symbol, timeframe, 35, 1, start=start, end=end)
    bandwalk40_1 = fs.i_bandwalk(symbol, timeframe, 40, 1, start=start, end=end)
    bandwalk45_1 = fs.i_bandwalk(symbol, timeframe, 45, 1, start=start, end=end)
    bandwalk50_1 = fs.i_bandwalk(symbol, timeframe, 50, 1, start=start, end=end)
    # 説明変数を正規化する。
    bandwalk5_1_mean = bandwalk5_1.mean()
    bandwalk5_1_std = bandwalk5_1.std()
    bandwalk10_1_mean = bandwalk10_1.mean()
    bandwalk10_1_std = bandwalk10_1.std()
    bandwalk15_1_mean = bandwalk15_1.mean()
    bandwalk15_1_std = bandwalk15_1.std()
    bandwalk20_1_mean = bandwalk20_1.mean()
    bandwalk20_1_std = bandwalk20_1.std()
    bandwalk25_1_mean = bandwalk25_1.mean()
    bandwalk25_1_std = bandwalk25_1.std()
    bandwalk30_1_mean = bandwalk30_1.mean()
    bandwalk30_1_std = bandwalk30_1.std()
    bandwalk35_1_mean = bandwalk35_1.mean()
    bandwalk35_1_std = bandwalk35_1.std()
    bandwalk40_1_mean = bandwalk40_1.mean()
    bandwalk40_1_std = bandwalk40_1.std()
    bandwalk45_1_mean = bandwalk45_1.mean()
    bandwalk45_1_std = bandwalk45_1.std()
    bandwalk50_1_mean = bandwalk50_1.mean()
    bandwalk50_1_std = bandwalk50_1.std()
    bandwalk5_1 = (bandwalk5_1 - bandwalk5_1_mean) / bandwalk5_1_std
    bandwalk10_1 = (bandwalk10_1 - bandwalk10_1_mean) / bandwalk10_1_std
    bandwalk15_1 = (bandwalk15_1 - bandwalk15_1_mean) / bandwalk15_1_std
    bandwalk20_1 = (bandwalk20_1 - bandwalk20_1_mean) / bandwalk20_1_std
    bandwalk25_1 = (bandwalk25_1 - bandwalk25_1_mean) / bandwalk25_1_std
    bandwalk30_1 = (bandwalk30_1 - bandwalk30_1_mean) / bandwalk30_1_std
    bandwalk35_1 = (bandwalk35_1 - bandwalk35_1_mean) / bandwalk35_1_std
    bandwalk40_1 = (bandwalk40_1 - bandwalk40_1_mean) / bandwalk40_1_std
    bandwalk45_1 = (bandwalk45_1 - bandwalk45_1_mean) / bandwalk45_1_std
    bandwalk50_1 = (bandwalk50_1 - bandwalk50_1_mean) / bandwalk50_1_std
    x = pd.concat(
        [bandwalk5_1, bandwalk10_1, bandwalk15_1, bandwalk20_1, bandwalk25_1,
        bandwalk30_1, bandwalk35_1, bandwalk40_1, bandwalk45_1, bandwalk50_1],
        axis=1)
    x = x.fillna(0.0)  # とりあえず0で埋める。
    x[(x==float("inf")) | (x==float("-inf"))] = 0.0  # とりあえず0で埋める。
    # 目的変数を格納する。
    roc0 = fs.i_roc(symbol, timeframe, 1, 0, start=start, end=end)
    roc0_mean = roc0.mean()
    roc0_std = roc0.std()
    roc0 = (roc0 - roc0_mean) / roc0_std
    y = roc0
    y = y.fillna(1)  # とりあえず1で埋める。
    y[(y==float("inf")) | (y==float("-inf"))] = 1  # とりあえず1で埋める。
    # モデルを作成する。
    if os.path.exists(folder_model) == False: os.makedirs(folder_model)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    path = (folder_model + "/" + symbol + str(timeframe) + "_" + start_str
        + "_" + end_str + ".pkl")
    if remove_model == 1 and os.path.exists(path) == True: os.remove(path)
    if os.path.exists(path) == True:
        model = joblib.load(path)
    else:
        model = linear_model.LinearRegression()
    model.fit(x, y)
    pred = model.predict(x)
    pred_mean = pred.mean()
    pred_std = pred.std()
    array = [pred_mean, pred_std, bandwalk5_1_mean, bandwalk5_1_std,
        bandwalk10_1_mean, bandwalk10_1_std, bandwalk15_1_mean,
        bandwalk15_1_std, bandwalk20_1_mean, bandwalk20_1_std,
        bandwalk25_1_mean, bandwalk25_1_std, bandwalk30_1_mean,
        bandwalk30_1_std, bandwalk35_1_mean, bandwalk35_1_std,
        bandwalk40_1_mean, bandwalk40_1_std, bandwalk45_1_mean,
        bandwalk45_1_std, bandwalk50_1_mean, bandwalk50_1_std]
    joblib.dump(model, path)
    return model, array
  
if __name__ == "__main__":
    argvs = sys.argv
    # バックテスト（ウォークフォワードテストを含む）の場合
    if argvs[1] == "backtest":
        backtest_start = time.time()
        parameter = THRESHOLD
        bounds = [
            (START_THRESHOLD, END_THRESHOLD)
        ]
        rranges = (
            # パラメータが1つだけのときは最後の「,」を忘れないこと
            slice(START_THRESHOLD, END_THRESHOLD, STEP_THRESHOLD),
        )
        fs = forex_system.ForexSystem()
        # ウォークフォワードテストの場合
        if WFT == 1:
            fs.walk_forward_test(
                strategy, parameter, SYMBOL, TIMEFRAME, POSITION, create_model,
                FOLDER_MODEL, REMOVE_MODEL, bounds, rranges, SPREAD, 1, DE,
                MIN_TRADE, MAXITER, POPSIZE, SEED, IN_SAMPLE_PERIOD,
                OUT_OF_SAMPLE_PERIOD, START_YEAR, START_MONTH, START_DAY,
                END_YEAR, END_MONTH, END_DAY)
        # バックテストの場合
        else:
            fs.backtest(
                strategy, parameter, SYMBOL, TIMEFRAME, POSITION, create_model,
                FOLDER_MODEL, REMOVE_MODEL, bounds, rranges, SPREAD,
                OPTIMIZATION, DE, MIN_TRADE, MAXITER, POPSIZE, SEED,
                START_YEAR, START_MONTH, START_DAY, END_YEAR, END_MONTH,
                END_DAY)
        backtest_end = time.time()
        # 結果を出力する。
        if backtest_end - backtest_start < 60.0:
            print(
                "バックテストの所要時間は",
                int(round(backtest_end - backtest_start)), "秒です。")
        else:
            print(
                "バックテストの所要時間は",
                int(round((backtest_end - backtest_start) / 60.0)), "分です。")
    # トレードの場合
    elif argvs[1] == "trade":
        print(
            "環境、アカウントID、アクセストークンの順で以下のように入力して下さい：")
        print(
            "practice "
            + "0123456 "
            + "0123456789abcdefghijklmnopqrstuv-"
            + "0123456789abcdefghijklmnopqrstuv")
        #argvs = input(">>> ").split()
        environment = "practice" #"live" #"sandbox" #"practice" # argvs[0]
        account_id = 2377909  # int(argvs[1])
        access_token = "de0b248d48367e62506f6f1430030a40-2f3471cc47949f542c2a60d392141b30"  #argvs[2]
        count = 100
        fs = forex_system.ForexSystem(environment, account_id, access_token)
        fs_model = forex_system.ForexSystem()
        parameter = THRESHOLD
        ticket = 0
        start = datetime(START_YEAR, START_MONTH, START_DAY)
        end = datetime(END_YEAR, END_MONTH, END_DAY)
        model, array = create_model(
            fs_model, SYMBOL, TIMEFRAME, FOLDER_MODEL, REMOVE_MODEL, start,
            end)
        pos = 0
        event = threading.Event()
        event.set()  # トレードの前に1回だけデータを取得する
        thread1 = threading.Thread(
            target=fs.update_data, args=(SYMBOL, TIMEFRAME, count, event))
        thread2 = threading.Thread(
            target=fs.trade, args=(strategy, parameter, SYMBOL, TIMEFRAME,
            POSITION, model, array, SPREAD, LOTS, EA, FOLDER_EA, FILE_EA,
            ticket, pos, event))
        thread1.start()
        time.sleep(10)  # 最初のデータ取得のため、少し時間を置く
        thread2.start()

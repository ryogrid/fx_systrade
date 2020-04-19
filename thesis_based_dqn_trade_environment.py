# coding:utf-8
import numpy as np
import scipy.sparse
import pickle
import talib as ta
from datetime import datetime as dt
import pytz
import os
import sys
import sklearn
import time
import random
from sklearn.preprocessing import StandardScaler
from collections import deque
import math
import copy

RATE_AND_DATE_STLIDE = int(5 / 5) # 5分足 #int(30 / 5) # 30分足
HALF_DAY_MODE = True # ageent側にも同じフラグがあって同期している必要があるので注意
USE_PAST_REWARD_FEATURES = True

ONE_YEAR_DAYS = 252
MONTH_DAYS = 21
TWO_MONTH_DAYS = 2 * MONTH_DAYS
THREE_MONTH_DAYS = 3 * MONTH_DAYS

# 与えられた価格のリストの最後の要素に対応する volatility(EMSD) となるスカラ値を返す
# len(partial_price_arr) は window_size と一致する必要がある
def calculate_volatility(partial_price_arr, window_size):
    alpha = 2 / float(window_size + 1)
    ema_arr = []
    emvar_arr = []
    delta = 0
    ema_arr.append(partial_price_arr[0])
    emvar_arr.append(delta)
    for idx in range(1, window_size):
        delta = partial_price_arr[idx] - ema_arr[idx - 1]
        ema_arr.append(ema_arr[idx - 1] + alpha * delta)
        emvar_arr.append((1 - alpha) * (emvar_arr[idx - 1] + alpha * delta * delta))

    print("calculate_volatility:" + str(len(partial_price_arr)))
    print("calculate_volatility:" + str(ema_arr[-1]))
    print("calculate_volatility:" + str(emvar_arr[-1]))
    emsd = math.sqrt(emvar_arr[-1])
    print("calculate_volatility:" + str(emsd))
    print("calculate_volatility: ----------------------------------")
    return emsd

class FXEnvironment:
    def __init__(self, train_data_num, time_series=32, holdable_positions=100, half_spread=0.0015, volatility_tgt = 0.1, bp = 0.000015):
        print("FXEnvironment class constructor called.")
        self.COMPETITION_TRAIN_DATA_NUM = train_data_num

        self.DATA_HEAD_ASOBI = 252 + 63 + 5 # MACDを算出するための期間（1年） + MACDを算出するための期間（63日 window） + 余裕を持たせる分
        if HALF_DAY_MODE:
            self.DATA_HEAD_ASOBI = 2 * self.DATA_HEAD_ASOBI

        self.tr_input_arr = None
        self.val_input_arr = None

        self.exchange_dates = []
        self.exchange_rates = []
        self.volatility_arr = []
        self.macd_arr = []

        self.time_series = time_series
        self.holdable_positions  = holdable_positions

        self.setup_serialized_fx_data()
        self.half_spread = half_spread

        self.volatility_tgt = volatility_tgt
        self.bp = bp

    def preprocess_data(self, X, scaler=None):
        if scaler == None:
            scaler = StandardScaler()
            scaler.fit(X)

        X_T = scaler.transform(X)
        return X_T, scaler

    def get_rsi(self, price_arr, cur_pos, period = 30):
        period_local = period
        if HALF_DAY_MODE:
            period_local = 2 * period_local
        if cur_pos <= period_local:
            return 0
        else:
            s = cur_pos - (period_local + 1)
        tmp_arr = price_arr[s:cur_pos]
        #tmp_arr.reverse()
        prices = np.array(tmp_arr, dtype=float)

        rsi_val = ta.RSI(prices, timeperiod = period_local)[-1]
        print("get_rsi:" + str(rsi_val))
        return rsi_val

    # def get_ma(self, price_arr, cur_pos, period=40):
    #     if cur_pos <= period:
    #         s = 0
    #     else:
    #         s = cur_pos - period
    #     tmp_arr = price_arr[s:cur_pos]
    #     tmp_arr.reverse()
    #     prices = np.array(tmp_arr, dtype=float)
    #
    #     return ta.SMA(prices, timeperiod = period)[-1]
    #
    # def get_ma_kairi(self, price_arr, cur_pos, period = None):
    #     ma = self.get_ma(price_arr, cur_pos)
    #     return ((price_arr[cur_pos] - ma) / ma) * 100.0
    #     return 0
    #
    # def get_bb_1(self, price_arr, cur_pos, period = 40):
    #     if cur_pos <= period:
    #         s = 0
    #     else:
    #         s = cur_pos - period
    #     tmp_arr = price_arr[s:cur_pos]
    #     tmp_arr.reverse()
    #     prices = np.array(tmp_arr, dtype=float)
    #
    #     return ta.BBANDS(prices, timeperiod = period)[0][-1]
    #
    # def get_bb_2(self, price_arr, cur_pos, period = 40):
    #     if cur_pos <= period:
    #         s = 0
    #     else:
    #         s = cur_pos - period
    #     tmp_arr = price_arr[s:cur_pos]
    #     tmp_arr.reverse()
    #     prices = np.array(tmp_arr, dtype=float)
    #
    #     return ta.BBANDS(prices, timeperiod = period)[2][-1]
    #
    # # periodは移動平均を求める幅なので20程度で良いはず...
    # def get_ema(self, price_arr, cur_pos, period = 20):
    #     if cur_pos <= period:
    #         s = 0
    #     else:
    #         s = cur_pos - period
    #     tmp_arr = price_arr[s:cur_pos]
    #     tmp_arr.reverse()
    #     prices = np.array(tmp_arr, dtype=float)
    #
    #     return ta.EMA(prices, timeperiod = period)[-1]
    #
    # def get_mo(self, price_arr, cur_pos, period=40):
    #     if cur_pos <= (period + 1):
    #         return 0
    #     else:
    #         s = cur_pos - (period + 1)
    #     tmp_arr = price_arr[s:cur_pos]
    #     tmp_arr.reverse()
    #     prices = np.array(tmp_arr, dtype=float)
    #
    #     return ta.CMO(prices, timeperiod = period)[-1]
    #
    # def get_po(self, price_arr, cur_pos, period=40):
    #     if cur_pos <= period:
    #         s = 0
    #     else:
    #         s = cur_pos - period
    #     tmp_arr = price_arr[s:cur_pos]
    #     tmp_arr.reverse()
    #     prices = np.array(tmp_arr, dtype=float)
    #
    #     return ta.PPO(prices)[-1]

    # def get_vorarity(self, price_arr, cur_pos, period = None):
    #     tmp_arr = []
    #     prev = -1.0
    #     for val in price_arr[cur_pos-period:cur_pos]:
    #         if prev == -1.0:
    #             tmp_arr.append(0.0)
    #         else:
    #             tmp_arr.append(val - prev)
    #         prev = val
    #
    #     return np.std(tmp_arr)

    def setup_macd_arr(self, price_arr, period = 63):
        local_period = period
        if HALF_DAY_MODE:
            local_period = 2 * local_period

        if HALF_DAY_MODE:
            local_ONE_YEAR_DAYS = 2 * ONE_YEAR_DAYS
        print("setup_macd_arr called")
        prices = np.array(price_arr, dtype=float)
        print(len(prices))
        fast = 8
        slow = 26
        if HALF_DAY_MODE:
            fast = 2 * fast
            slow = 2 * slow

        macd, macdsignal, macdhist = ta.MACD(prices, fastperiod=fast, slowperiod=slow)
        print(len(macd))
        self.macd_arr = macd
        # まず period 期間の標準偏差で割ることでリストの値を q{t} に置き換える
        for idx in range(local_period, len(self.macd_arr)):
            price_period_std = np.std(price_arr[idx - local_period + 1:idx + 1])
            self.macd_arr[idx] = self.macd_arr[idx] / price_period_std
        # 次に1年間の標準偏差で割ることでリストの値を論文通りのMACD{i}に置き換える
        for idx in range(local_ONE_YEAR_DAYS, len(self.macd_arr)):
            price_year_std = np.std(price_arr[idx - ONE_YEAR_DAYS + 1:idx + 1])
            self.macd_arr[idx] = self.macd_arr[idx] / price_year_std

    # def get_macd(self, price_arr, cur_pos, period = 63):
    #     if cur_pos <= period:
    #         s = 0
    #     else:
    #         s = cur_pos - period
    #     tmp_arr = price_arr[s:cur_pos]
    #     #tmp_arr.reverse()
    #     prices = np.array(tmp_arr, dtype=float)
    #
    #     macd, macdsignal, macdhist = ta.MACD(prices, fastperiod=8, slowperiod=24)
    #
    #     print("get_macd:" + str(macd[-1]))
    #     return macd[-1]

    def get_macd(self, price_arr, cur_pos):
        return self.macd_arr[cur_pos]

    # 日本時間で土曜7:00-月曜7:00までは取引不可として元データから取り除く
    # なお、本来は月曜朝5:00から取引できるのが一般的なようである
    def is_weekend(self, date_str):
        tz = pytz.timezone('Asia/Tokyo')
        dstr = date_str.replace(".","-")
        tdatetime = dt.strptime(dstr, '%Y-%m-%d %H:%M:%S')
        tz_time = tz.localize(tdatetime)
        gmt_plus2_tz = pytz.timezone('Etc/GMT+2')
        gmt_plus2_time = tz_time.astimezone(gmt_plus2_tz)
        week = gmt_plus2_time.weekday()
        return (week == 5 or week == 6)

    def logfile_writeln_with_fd(self, out_fd, log_str):
        out_fd.write(log_str + "\n")
        #out_fd.flush()

    def make_serialized_data(self, start_idx, end_idx, step, x_arr_fpath):
        input_mat = []
        print("all rate and data size: " + str(len(self.exchange_rates)))
        for i in range(start_idx, end_idx, step):
            if i % 100 == 0:
                print("current date idx: " + str(i))
            input_mat.append(
                [self.exchange_rates[i],
                 self.get_rsi(self.exchange_rates, i),
                 self.get_macd(self.exchange_rates, i),
                 ]
            )

        input_mat = np.array(input_mat, dtype=np.float64)
        with open(x_arr_fpath, 'wb') as f:
            pickle.dump(input_mat, f)

        return input_mat

    # calculate EMSD(Exponentially wighted moving standard deviation)
    def setup_volatility_arr(self, rate_arr, window_size):
        local_window_size = window_size
        if HALF_DAY_MODE:
            local_window_size = 2 * local_window_size
        for idx in range(len(rate_arr)):
            if idx + 1 < local_window_size:
                self.volatility_arr.append(0)
            else:
                s = (idx + 1) - local_window_size
                tmp_arr = rate_arr[s:idx + 1]
                self.volatility_arr.append(calculate_volatility(tmp_arr, local_window_size))

                # s = (idx + 1) - window_size
                # tmp_arr = rate_arr[s:idx + 1]
                # print("get_volatility_arr:" + str(len(tmp_arr)))
                # prices = np.array(tmp_arr, dtype=float)
                # ema_arr = ta.EMA(prices, timeperiod = window_size)
                # emvar_arr = []
                # for sd_idx in range(len(ema_arr)):
                #     if sd_idx == 0:
                #         emvar_arr.append(0.0)
                #     else:
                #         delta = self.exchange_rates[idx] - ema_arr[sd_idx - 1]
                #         emvar_arr.append((1 - alpha) * (emvar_arr[sd_idx - 1] + alpha * delta * delta))
                # self.volatility_arr.append(math.sqrt(emvar_arr[-1]))


    def setup_serialized_fx_data(self):
        if False: #self.is_fist_call == False and os.path.exists("./exchange_rates.pickle"):
            with open("./exchange_dates.pickle", 'rb') as f:
                self.exchange_dates = pickle.load(f)
            with open("./exchange_rates.pickle", 'rb') as f:
                self.exchange_rates = pickle.load(f)
        else:
            rates_fd = open('./USD_JPY_2001_2008_5min.csv', 'r')
            leg_split_symbol_str = "23:55:00"
            additional_symbol = "xxxxx" # 基本的には存在しない文字列を指定しておく
            if HALF_DAY_MODE:
                additional_symbol = "11:55:00"

            # 日足のデータを得る (HALF_DAY_MODEの時は半日足のデータを得る)
            for line in rates_fd:
                splited = line.split(",")
                if (leg_split_symbol_str in splited[0] or additional_symbol in splited[0]) and splited[2] != "High" and splited[0] != "<DTYYYYMMDD>" and splited[0] != "204/04/26" and splited[
                    0] != "20004/04/26" and self.is_weekend(splited[0]) == False:
                    time = splited[0].replace("/", "-")  # + " " + splited[1]
                    val = float(splited[4]) # close price
                    self.exchange_dates.append(time)
                    self.exchange_rates.append(val)
            # 足の長さを調節する
            self.exchange_dates = self.exchange_dates[::RATE_AND_DATE_STLIDE]
            self.exchange_rates = self.exchange_rates[::RATE_AND_DATE_STLIDE]
            with open("./exchange_rates.pickle", 'wb') as f:
                pickle.dump(self.exchange_rates, f)
            with open("./exchange_dates.pickle", 'wb') as f:
                pickle.dump(self.exchange_dates, f)

        print("data size of all rates for train and test: " + str(len(self.exchange_rates)))

        if False:  # self.is_fist_call == False and os.path.exists("./volatility_arr.pickle"):
            with open('./volatility_arr.pickle', 'rb') as f:
                self.volatility_arr = pickle.load(f)
        else:
            # setup self.volatility_arr
            # 60day window
            window_size = 60
            self.setup_volatility_arr(self.exchange_rates, window_size)
            with open("./volatility_arr.pickle", 'wb') as f:
                pickle.dump(self.volatility_arr, f)

        if False:  # self.is_fist_call == False and os.path.exists("./macd_arr.pickle"):
            with open('./macd_arr.pickle', 'rb') as f:
                self.macd_arr = pickle.load(f)
        else:
            self.setup_macd_arr(self.exchange_rates, period = 63)
            with open("./macd_arr.pickle", 'wb') as f:
                pickle.dump(self.macd_arr, f)

        if False: #self.is_fist_call == False and os.path.exists("./all_input_mat.pickle"):
            with open('./all_input_mat.pickle', 'rb') as f:
                all_input_mat = pickle.load(f)
        else:
            all_input_mat = \
                self.make_serialized_data(self.DATA_HEAD_ASOBI, len(self.exchange_rates) - self.DATA_HEAD_ASOBI, 1, './all_input_mat.pickle')

        # TODO: 論文では価格のみ normalize したとあるが、面倒なので全ての特徴量を normalize してしまう
        self.tr_input_arr, tr_scaler = self.preprocess_data(all_input_mat[0:self.COMPETITION_TRAIN_DATA_NUM])
        self.ts_input_arr, _ =  self.preprocess_data(all_input_mat[self.COMPETITION_TRAIN_DATA_NUM:], tr_scaler)

        print("input features sets for tarin: " + str(self.COMPETITION_TRAIN_DATA_NUM))
        print("input features sets for test: " + str(len(self.ts_input_arr)))
        print("finished setup environment data.")

    # type_str: "train", "test"
    def get_env(self, type_str):
        if(type_str == "backtest"):
            return self.InnerFXEnvironment(self.tr_input_arr, self.exchange_dates, self.exchange_rates,
                                           self.DATA_HEAD_ASOBI, holdable_positions = self.holdable_positions, half_spread=self.half_spread,
                                           time_series = self.time_series, volatility_tgt = self.volatility_tgt, is_backtest=True)
        if(type_str == "auto_backtest"):
            return self.InnerFXEnvironment(self.tr_input_arr, self.exchange_dates, self.exchange_rates,
                                           self.DATA_HEAD_ASOBI, holdable_positions = self.holdable_positions, half_spread=self.half_spread,
                                           time_series = self.time_series, volatility_tgt = self.volatility_tgt, is_backtest=True, is_auto_backtest=True)
        elif(type_str == "backtest_test"):
            return self.InnerFXEnvironment(self.ts_input_arr, self.exchange_dates, self.exchange_rates,
                                           self.DATA_HEAD_ASOBI + self.COMPETITION_TRAIN_DATA_NUM, holdable_positions = self.holdable_positions, half_spread=self.half_spread,
                                           time_series = self.time_series, volatility_tgt = self.volatility_tgt, is_backtest=True)
        elif(type_str == "auto_backtest_test"):
            return self.InnerFXEnvironment(self.ts_input_arr, self.exchange_dates, self.exchange_rates,
                                           self.DATA_HEAD_ASOBI + self.COMPETITION_TRAIN_DATA_NUM, holdable_positions = self.holdable_positions, half_spread=self.half_spread,
                                           time_series = self.time_series, volatility_tgt = self.volatility_tgt, is_backtest=True, is_auto_backtest = True)
        else:
            return self.InnerFXEnvironment(self.tr_input_arr, self.exchange_dates, self.exchange_rates,
                                           self.DATA_HEAD_ASOBI, volatility_arr = self.volatility_arr, time_series = self.time_series, holdable_positions = self.holdable_positions, half_spread=self.half_spread,
                                           volatility_tgt = self.volatility_tgt, is_backtest=False)

    class InnerFXEnvironment:
        def __init__(self, input_arr, exchange_dates, exchange_rates, idx_geta, volatility_arr = None, time_series=32, half_spread=0.0015, holdable_positions=100, is_backtest = False, volatility_tgt = 0.1, bp = 0.000015, is_auto_backtest = False):
            self.LONG = 0
            self.SHORT = 1
            self.NOT_HAVE = 2

            self.input_arr = input_arr
            self.input_arr_len = len(input_arr)
            self.exchange_dates = exchange_dates
            self.exchange_rates = exchange_rates
            self.half_spread = half_spread
            self.time_series = time_series
            self.initial_cur_idx_val = (time_series - 1) #LSTMで過去のstateを見るため、その分はずらしてスタートする
            self.cur_idx = self.initial_cur_idx_val
            self.idx_geta = idx_geta
            self.is_backtest = is_backtest
            self.is_auto_backtest = is_auto_backtest
            self.volatility_tgt = volatility_tgt

            self.volatility_arr = volatility_arr

            # インデックスは idx_getaを加算しないcur_idx基準
            # 1イテレーションの間の分のみ保持する
            self.past_return_arr = [0.0] * (self.input_arr_len + 10)
            # 過去のreturnから算出される値を特徴量として加えるために用いる
            # past_return_arr とインデックスの基準は同じ
            # 1イテレーションの間の分のみ保持する
            self.past_return_emsd_arr = [0.0] * (self.input_arr_len + 10)

            # reward の計算にはbpをトランザクションコストのレートとして用いるが、実際の取引でのスプレッドは half_sparedを用いる
            self.bp = bp
            # 常に 5000ドル単位で売買を行う（保有可能ポジションは 1）
            # 初期資産100万円で、半分の50万円程度を用いる前提で、1ドル100円のレートを想定して設定した
            self.fixed_open_currency_num_mue = 5000

            if self.is_backtest and self.is_auto_backtest:
                self.log_fd_bt = open("./auto_backtest_log_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv", mode = "w")
            elif self.is_backtest:
                self.log_fd_bt = open("./backtest_log_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt", mode="w")
            else:
                self.log_fd_bt = open("./learn_trade_log_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt",
                                      mode="w")

            self.start = time.time()
            self.idx_step = 1

            self.done = False
            self.holdable_positions = holdable_positions

            self.portfolio_mngr = PortforioManager(exchange_rates, self.half_spread, holdable_position_num = self.holdable_positions, fixed_open_currency_num_mue=self.fixed_open_currency_num_mue)

            self.action_t_minus_one = 0
            self.action_t_minus_two = 0

        def get_rand_str(self):
            return str(random.randint(0, 10000000))

        def logfile_writeln_bt(self, log_str):
            self.log_fd_bt.write(log_str + "\n")
            self.log_fd_bt.flush()

        def close_all(self, cur_episode_rate_idx):
            won_pips, won_money = self.portfolio_mngr.close_all(cur_episode_rate_idx)

            return won_pips, won_money

        def caluculate_reward(self, cur_episode_rate_idx):
            p_t = self.exchange_rates[cur_episode_rate_idx]
            p_t_minus_one = self.exchange_rates[cur_episode_rate_idx - 1]
            r_t = p_t - p_t_minus_one
            volatility_t_minus_one = self.volatility_arr[cur_episode_rate_idx - 1]
            volatility_t_minus_two = self.volatility_arr[cur_episode_rate_idx - 2]
            evaluated_position_member = self.action_t_minus_one * (self.volatility_tgt / volatility_t_minus_one) * r_t
            transaction_cost_member_child = (self.volatility_tgt / volatility_t_minus_one) * self.action_t_minus_one - (self.volatility_tgt / volatility_t_minus_two) * self.action_t_minus_two
            transaction_cost_member_parent = self.bp * p_t_minus_one * abs(transaction_cost_member_child)
            reward = 1.0 * (evaluated_position_member - transaction_cost_member_parent)

            return reward

        def step(self, action_num):
            cur_episode_rate_idx = self.idx_geta + self.cur_idx
            did_close = False

            won_money = 0

            if action_num == -1:
                action = "SELL"
            elif action_num == 0:
                action = "DONOT"
            elif action_num == 1:
                 action = "BUY"
            else:
                raise Exception(str(action_num) + " is invalid.")

            a_log_str_line = "log," + str(self.cur_idx) + "," + action

            # rewardは現在のactionではなく、過去の2つのアクションによって決まるので
            # 個別に求める必要はない
            if self.is_backtest:
                # バックテストを行う際はrewardは用いないため算出不要
                reward = 0
            else:
                reward = self.caluculate_reward(cur_episode_rate_idx)
                print("reward_at_step," + str(self.cur_idx)  + "," + str(reward))

            # 過去の2アクションを保持しておく必要があるため入れ替えを行う
            self.action_t_minus_two = self.action_t_minus_one
            self.action_t_minus_one = action_num

            if action == "BUY":
                if self.portfolio_mngr.having_long_or_short == self.NOT_HAVE:
                    buy_val = self.portfolio_mngr.buy(cur_episode_rate_idx)
                    a_log_str_line += ",OPEN_LONG" + ",0,0," + str(
                    self.exchange_rates[cur_episode_rate_idx]) + "," + str(buy_val)
                elif self.portfolio_mngr.having_long_or_short == self.SHORT:
                    #持っているSHORTポジションをクローズ
                    _ , won_money = self.close_all(cur_episode_rate_idx)
                    # LONGに持ちかえる
                    buy_val = self.portfolio_mngr.buy(cur_episode_rate_idx)
                    a_log_str_line += ",CLOSE_SHORT_AND_OPEN_LONG" + ",0,0," + str(self.exchange_rates[cur_episode_rate_idx]) + "," + str(buy_val)
                    did_close = True
                elif self.portfolio_mngr.having_long_or_short == self.LONG:
                    #既にLONGポジションを持っていたら何もしない
                    a_log_str_line += ",HOLD_LONG" + ",0,0," + str(self.exchange_rates[cur_episode_rate_idx]) + ",0"
                else:
                    raise Exception("unknown state")
            elif action == "DONOT":
                if self.portfolio_mngr.having_long_or_short == self.LONG:
                    operation_str = ",POSITION_HOLD_LONG,0,"
                elif self.portfolio_mngr.having_long_or_short == self.SHORT:
                    operation_str = ",POSITION_HOLD_SHORT,0,"
                elif self.portfolio_mngr.having_long_or_short == self.NOT_HAVE: #DONOTのダミーポジションだけ存在する場合
                    operation_str = ",KEEP_NO_POSITION,0,"

                if self.portfolio_mngr.having_long_or_short == self.NOT_HAVE:
                    diff_str = "0"
                else:
                    diff_str = str(self.portfolio_mngr.get_evaluated_val_diff_of_all_pos(cur_episode_rate_idx))

                a_log_str_line += operation_str + diff_str + "," + str(
                    self.exchange_rates[cur_episode_rate_idx]) + ",0"
            elif action == "SELL":
                if self.portfolio_mngr.having_long_or_short == self.NOT_HAVE:
                    sell_val = self.portfolio_mngr.sell(cur_episode_rate_idx)
                    a_log_str_line += ",OPEN_SHORT" + ",0,0," + str(
                    self.exchange_rates[cur_episode_rate_idx]) + "," + str(sell_val)
                elif self.portfolio_mngr.having_long_or_short == self.LONG:
                    #持っているLONGポジションをクローズ
                    _ , won_money = self.close_all(cur_episode_rate_idx)
                    # SHORTに持ちかえる
                    sell_val = self.portfolio_mngr.sell(cur_episode_rate_idx)
                    a_log_str_line += ",CLOSE_LONG_AND_OPEN_SHORT" + ",0,0," + str(self.exchange_rates[cur_episode_rate_idx]) + "," + str(sell_val)
                    did_close = True
                elif self.portfolio_mngr.having_long_or_short == self.SHORT:
                    #既にSHORTポジションを持っていたら何もしない
                    a_log_str_line += ",HOLD_SHORT" + ",0,0," + str(self.exchange_rates[cur_episode_rate_idx]) + ",0"
                else:
                    raise Exception("unknown state")
            else:
                raise Exception(str(action) + " is invalid.")

            a_log_str_line += "," + str(self.portfolio_mngr.get_current_portfolio(cur_episode_rate_idx)) +\
                              "," + str(self.portfolio_mngr.total_won_pips) + "," + str(self.portfolio_mngr.having_money) + "," + str(self.portfolio_mngr.get_position_num())


            if self.is_auto_backtest:
                #自動バックテストの際は、CLOSEのアクションの内容だけ出力させる
                if did_close:
                    self.logfile_writeln_bt(a_log_str_line)
            else:
                self.logfile_writeln_bt(a_log_str_line)



            # 過去のreturnから求まる特徴量算出のための値を用意しておく
            if USE_PAST_REWARD_FEATURES:
                # このstepでのリターンを記録しておく
                self.past_return_arr[self.cur_idx] = won_money
                # この episodeでの 60-day span の　EMSD を埋めておく
                day_span = 60
                if HALF_DAY_MODE:
                    day_span = 2 * day_span

                # この後、stepメソッドが返すstateは self.cur_idx + 1 のものなので、そこを埋める
                if self.cur_idx >= day_span + self.initial_cur_idx_val:
                    #必要な要素数が用意できる場合のみ設定する. 設定しない場合は 0.0 で初期化されているので問題ない
                    # +1　しているのは self.cur_idx の要素が slice したデータに含まれるようにするため
                    self.past_return_emsd_arr[self.cur_idx + 1] = \
                        calculate_volatility(self.past_return_arr[self.cur_idx - day_span + 1:self.cur_idx + 1], day_span)

            self.cur_idx += self.idx_step
            #if (self.cur_idx) >= (len(self.input_arr) - (self.time_series - 1) - 1):
            if (self.cur_idx) >= len(self.input_arr):
                self.logfile_writeln_bt("finished backtest.")
                print("finished backtest.")
                process_time = time.time() - self.start
                self.logfile_writeln_bt("excecution time of backtest: " + str(process_time))
                self.logfile_writeln_bt("result of portfolio: " + str(self.portfolio_mngr.get_current_portfolio(cur_episode_rate_idx)))
                print("result of portfolio: " + str(self.portfolio_mngr.get_current_portfolio(cur_episode_rate_idx)))
                self.log_fd_bt.flush()
                self.log_fd_bt.close()
                return None, reward, True


            if USE_PAST_REWARD_FEATURES:
                # 過去のreturnから求まる特徴量を next_state に加える
                next_state_tmp = self.input_arr[self.cur_idx - self.time_series + 1:self.cur_idx + 1]
                local_ONE_YEAR_DAYS = 2 * ONE_YEAR_DAYS if HALF_DAY_MODE else ONE_YEAR_DAYS
                local_MONTH_DAYS = 2 * MONTH_DAYS if HALF_DAY_MODE else MONTH_DAYS
                local_TWO_MONTH_DAYS = 2 * TWO_MONTH_DAYS if HALF_DAY_MODE else TWO_MONTH_DAYS
                local_THREE_MONTH_DAYS = 2 * THREE_MONTH_DAYS if HALF_DAY_MODE else THREE_MONTH_DAYS

                one_year_sqrt = math.sqrt(local_ONE_YEAR_DAYS)
                one_month_sqrt = math.sqrt(local_MONTH_DAYS)
                two_month_sqrt = math.sqrt(local_TWO_MONTH_DAYS)
                three_month_sqrt = math.sqrt(local_THREE_MONTH_DAYS)

                next_state = np.array([])
                #print("------------------------------")
                for idx, a_feature_list in enumerate(next_state_tmp):
                    #print(a_feature_list)
                    daily_normalized_1year_return = -0.5
                    daily_normalized_1month_return = -0.5
                    daily_normalized_2month_return = -0.5
                    daily_normalized_3month_return = -0.5

                    # self.cur_idx は既に +1 インクリメントされているので 比較に等号は含まない
                    # スライスする場合も self.cur_idx要素は含めずに計算するので +1 はしない
                    if self.cur_idx - self.time_series + idx > local_ONE_YEAR_DAYS + self.initial_cur_idx_val:
                        daily_normalized_1year_return = np.sum(self.past_return_arr[self.cur_idx - local_ONE_YEAR_DAYS - self.time_series + idx:self.cur_idx - self.time_series + idx]) / \
                                                           (self.past_return_emsd_arr[self.cur_idx - 1 - self.time_series + idx] * one_year_sqrt + 0.0001)
                    if self.cur_idx - self.time_series + idx > local_MONTH_DAYS + self.initial_cur_idx_val:
                        daily_normalized_1month_return = np.sum(self.past_return_arr[self.cur_idx - local_MONTH_DAYS - self.time_series + idx:self.cur_idx  - self.time_series + idx]) / \
                                                        (self.past_return_emsd_arr[self.cur_idx - 1 - self.time_series + idx] * one_month_sqrt + 0.0001)
                    if self.cur_idx - self.time_series + idx > local_TWO_MONTH_DAYS + self.initial_cur_idx_val:
                        daily_normalized_2month_return = np.sum(self.past_return_arr[self.cur_idx - local_TWO_MONTH_DAYS - self.time_series + idx:self.cur_idx - self.time_series + idx]) / \
                                                        (self.past_return_emsd_arr[self.cur_idx - 1 - self.time_series + idx] * two_month_sqrt + 0.0001)
                    if self.cur_idx - self.time_series + idx > local_THREE_MONTH_DAYS + self.initial_cur_idx_val:
                        daily_normalized_3month_return = np.sum(self.past_return_arr[self.cur_idx - local_THREE_MONTH_DAYS - self.time_series + idx:self.cur_idx - self.time_series + idx]) / \
                                                        (self.past_return_emsd_arr[self.cur_idx - 1 - self.time_series + idx] * three_month_sqrt + 0.0001)

                    new_list = np.append(a_feature_list, daily_normalized_1year_return)
                    new_list = np.append(new_list, daily_normalized_1month_return)
                    new_list = np.append(new_list, daily_normalized_2month_return)
                    new_list = np.append(new_list, daily_normalized_3month_return)
                    #print(new_list)
                    next_state = np.append(next_state, new_list)

                    print("calculate_return_features," + str(daily_normalized_1year_return) + "," + str(daily_normalized_1month_return) + "," + \
                            str(daily_normalized_2month_return) + "," + str(daily_normalized_3month_return))
            else:
                next_state = self.input_arr[self.cur_idx - self.time_series + 1:self.cur_idx + 1]


            # 第四返り値はエピソードの識別子を格納するリスト. 第0要素は返却する要素に対応するもので、
            # それ以外の要素がある場合は、close時にさかのぼって エピソードのrewardを更新するためのもの
            return next_state, reward, False

class PortforioManager:

    def __init__(self, exchange_rates, half_spred=0.0015, holdable_position_num = 100, fixed_open_currency_num_mue = 5000, is_backtest=False):
        self.holdable_position_num = holdable_position_num
        self.exchange_rates = exchange_rates
        self.half_spread = half_spred
        self.is_backtest = is_backtest

        self.LONG = 0
        self.SHORT = 1
        self.NOT_HAVE = 2

        self.having_money = 1000000.0
        # 常に fixed_open_currency_num_mue 数のドル単位で売買を行う（保有可能ポジションは 1）
        self.fixed_open_currency_num_mue = fixed_open_currency_num_mue
        self.total_won_pips = 0.0

        # 各要素は [購入時の価格（スプレッド含む）, self.LONG or self.SHORT, 数量]
        # 数量は通貨の数を表しLONGであれば正、SHORTであれば負の値となる
        self.positions = []
        self.position_num = 0
        self.donot_num = 0

        # ポジションは複数持つが、一種類のものしか持たないという制約を設けるため
        # 判別が簡単になるようにこのフィールドを設ける
        self.having_long_or_short = self.NOT_HAVE

    def get_position_num(self):
        return self.position_num

    def additional_pos_openable(self):
        return self.position_num < self.holdable_position_num

    # 規約: 保持可能なポジション数を超える場合は呼び出されない
    # ロングポジションを最大保持可能数における1単位分購入する（購入通貨数が整数になるような調整は行わない）
    def buy(self, rate_idx):
        pos_kind = self.LONG
        trade_val = self.exchange_rates[rate_idx] + self.half_spread
        # currency_num = (self.fixed_use_money_mue / (self.holdable_position_num - self.position_num)) / trade_val
        currency_num = self.fixed_open_currency_num_mue

        self.positions.append([trade_val, pos_kind, currency_num, rate_idx])
        self.position_num += 1
        self.having_money -= trade_val * currency_num
        self.having_long_or_short = self.LONG
        return trade_val

    # 規約: 保持可能なポジション数を超える場合は呼び出されない
    # ショートポジションを最大保持可能数における1単位分購入する（購入通貨数が整数になるような調整は行わない）
    def sell(self, rate_idx):
        pos_kind = self.SHORT
        trade_val = self.exchange_rates[rate_idx] - self.half_spread
        #currency_num = (self.fixed_use_money_mue / (self.holdable_position_num - self.position_num)) / trade_val
        currency_num = self.fixed_open_currency_num_mue

        self.positions.append([trade_val, pos_kind, currency_num, rate_idx])
        self.position_num += 1
        self.having_money -= trade_val * currency_num
        self.having_long_or_short = self.SHORT
        return trade_val

    # 指定された一単位のポジションをクローズする（1単位は1通貨を意味するものではない）
    def close_long(self, position, rate_idx):
        # 保持しているロングポジションをクローズする
        cur_price = self.exchange_rates[rate_idx] - self.half_spread
        trade_result = position[2] * cur_price - position[2] * position[0]
        return_money = position[2] * cur_price
        won_pips = cur_price - position[0]

        return won_pips, trade_result, return_money

    # 指定された一単位のポジションをクローズする（1単位は1通貨を意味するものではない）
    def close_short(self, position, rate_idx):
        # 保持しているショートポジションをクローズする
        cur_price = self.exchange_rates[rate_idx] + self.half_spread
        trade_result = position[2] * position[0] - position[2] * cur_price
        won_pips = position[0] - cur_price

        # 買い物度↓した時に得られる損益をオープン時に利用した証拠金に足し合わせることで評価
        # 1)まず損益を求める
        diff = position[2] * (position[0] - cur_price)
        # 2)オープン時にとられた証拠金を求める
        collateral_at_open = position[0] * position[2]
        # 3)2つを足し合わせた額が決済によって戻ってくるお金
        return_money = collateral_at_open + diff

        return won_pips, trade_result, return_money

    # 全てのポジションをcloseしてしまう
    # ポジションの種類が混在していても呼び出し方法は変える必要がない
    def close_all(self, rate_idx):
        won_pips_sum = 0
        won_money_sum = 0
        returned_money_sum = 0
        for position in self.positions:
            if position[1] == self.LONG:
                won_pips, won_money, return_money = self.close_long(position, rate_idx)
            elif position[1] == self.SHORT: # self.SHORT
                won_pips, won_money, return_money = self.close_short(position, rate_idx)

            won_pips_sum += won_pips
            won_money_sum += won_money
            returned_money_sum += return_money


        self.having_long_or_short = self.NOT_HAVE
        self.total_won_pips += won_pips_sum
        self.having_money += returned_money_sum
        self.positions = []
        self.position_num = 0

        return won_pips_sum, won_money_sum

    # 現在のpipsで見た保有ポジションの評価損益
    def get_evaluated_val_diff_of_all_pos(self, rate_idx):
        total_evaluated_money_diff = 0
        total_currecy_num = 0
        cur_price_no_spread = self.exchange_rates[rate_idx]
        for position in self.positions:
            if position[1] == self.LONG:
                total_evaluated_money_diff += position[2] * ((cur_price_no_spread - self.half_spread) - position[0])
            else: # self.SHORT
                total_evaluated_money_diff += position[2] * (position[0] - (cur_price_no_spread + self.half_spread))
            total_currecy_num +=  position[2]

        if total_currecy_num == 0:
            return 0
        else:
            return total_evaluated_money_diff / total_currecy_num

    def get_current_portfolio(self, rate_idx):
        total_evaluated_money = 0
        cur_price_no_spread = self.exchange_rates[rate_idx]
        for position in self.positions:
            if position[1] == self.LONG:
                # 売った際に得られる現金で評価
                total_evaluated_money += position[2] * (cur_price_no_spread - self.half_spread)
            elif position[1] == self.SHORT:
                # 買い物度↓した時に得られる損益をオープン時に利用した証拠金に足し合わせることで評価
                # 1)まず損益を求める
                diff = position[2] * (position[0] - (cur_price_no_spread + self.half_spread))
                # 2)オープン時にとられた証拠金を求める
                collateral_at_open = position[0] * position[2]
                # 3)2つを足し合わせた額が決済によって戻ってくるお金
                total_evaluated_money += collateral_at_open + diff

        return self.having_money + total_evaluated_money

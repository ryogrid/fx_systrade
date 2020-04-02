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

RATE_AND_DATE_STLIDE = int(5 / 5) # 5分足 #int(30 / 5) # 30分足

class FXEnvironment:
    def __init__(self, train_data_num, time_series=32, holdable_positions=100, predict_future_legs=10, half_spread=0.0015):
        print("FXEnvironment class constructor called.")
        self.INPUT_LEN = 1
        self.predict_future_legs = predict_future_legs
        self.COMPETITION_TRAIN_DATA_NUM = train_data_num
        self.half_spread = half_spread

        self.CHART_TYPE_JDG_LEN = 40
        self.DATA_HEAD_ASOBI = 200

        self.FEATURE_NAMES = ["current_rate", "diff_ratio_between_previous_rate", "rsi", "ma", "ma_kairi", "bb_1",
                              "bb_2", "mo", "vorariity", "chart_type"]
        self.tr_input_arr = None
        self.tr_angle_arr = None
        self.val_input_arr = None
        self.val_angle_arr = None

        self.exchange_dates = None
        self.exchange_rates = None
        self.reverse_exchange_rates = None

        self.time_series = time_series
        self.holdable_positions  = holdable_positions

        self.setup_serialized_fx_data()

    def preprocess_data(self, X, scaler=None):
        if scaler == None:
            scaler = StandardScaler()
            scaler.fit(X)

        X_T = scaler.transform(X)
        return X_T, scaler

    # 0->flat 1->upper line 2-> downer line 3->above is top 4->below is top
    def judge_chart_type(self, data_arr):
        max_val = 0
        min_val = float("inf")

        last_idx = len(data_arr)-1

        for idx in range(len(data_arr)):
            if data_arr[idx] > max_val:
                max_val = data_arr[idx]
                max_idx = idx

            if data_arr[idx] < min_val:
                min_val = data_arr[idx]
                min_idx = idx


        if max_val == min_val:
            return 0

        if min_idx == 0 and max_idx == last_idx:
            return 1

        if max_idx == 0 and min_idx == last_idx:
            return 2

        if max_idx != 0 and max_idx != last_idx and min_idx != 0 and min_idx != last_idx:
            return 0

        if max_idx != 0 and max_idx != last_idx:
            return 3

        if min_idx != 0 and min_idx != last_idx:
            return 4

        return 0

    def get_rsi(self, price_arr, cur_pos, period = 40):
        if cur_pos <= period:
    #        s = 0
            return 0
        else:
            s = cur_pos - (period + 1)
        tmp_arr = price_arr[s:cur_pos]
        tmp_arr.reverse()
        prices = np.array(tmp_arr, dtype=float)

        return ta.RSI(prices, timeperiod = period)[-1]


    #def get_ma(self, price_arr, cur_pos, period = 20):
    def get_ma(self, price_arr, cur_pos, period=40):
        if cur_pos <= period:
            s = 0
        else:
            s = cur_pos - period
        tmp_arr = price_arr[s:cur_pos]
        tmp_arr.reverse()
        prices = np.array(tmp_arr, dtype=float)

        return ta.SMA(prices, timeperiod = period)[-1]

    def get_ma_kairi(self, price_arr, cur_pos, period = None):
        ma = self.get_ma(price_arr, cur_pos)
        return ((price_arr[cur_pos] - ma) / ma) * 100.0
        return 0

    def get_bb_1(self, price_arr, cur_pos, period = 40):
        if cur_pos <= period:
            s = 0
        else:
            s = cur_pos - period
        tmp_arr = price_arr[s:cur_pos]
        tmp_arr.reverse()
        prices = np.array(tmp_arr, dtype=float)

        return ta.BBANDS(prices, timeperiod = period)[0][-1]

    def get_bb_2(self, price_arr, cur_pos, period = 40):
        if cur_pos <= period:
            s = 0
        else:
            s = cur_pos - period
        tmp_arr = price_arr[s:cur_pos]
        tmp_arr.reverse()
        prices = np.array(tmp_arr, dtype=float)

        return ta.BBANDS(prices, timeperiod = period)[2][-1]

    # periodは移動平均を求める幅なので20程度で良いはず...
    def get_ema(self, price_arr, cur_pos, period = 20):
        if cur_pos <= period:
            s = 0
        else:
            s = cur_pos - period
        tmp_arr = price_arr[s:cur_pos]
        tmp_arr.reverse()
        prices = np.array(tmp_arr, dtype=float)

        return ta.EMA(prices, timeperiod = period)[-1]


    # def get_ema_rsi(price_arr, cur_pos, period = None):
    #     return 0

    # def get_cci(self, price_arr, cur_pos, period = None):
    #     return 0

    #def get_mo(self, price_arr, cur_pos, period = 20):
    def get_mo(self, price_arr, cur_pos, period=40):
        if cur_pos <= (period + 1):
    #        s = 0
            return 0
        else:
            s = cur_pos - (period + 1)
        tmp_arr = price_arr[s:cur_pos]
        tmp_arr.reverse()
        prices = np.array(tmp_arr, dtype=float)

        return ta.CMO(prices, timeperiod = period)[-1]

    #def get_po(self, price_arr, cur_pos, period = 10):
    def get_po(self, price_arr, cur_pos, period=40):
        if cur_pos <= period:
            s = 0
        else:
            s = cur_pos - period
        tmp_arr = price_arr[s:cur_pos]
        tmp_arr.reverse()
        prices = np.array(tmp_arr, dtype=float)

        return ta.PPO(prices)[-1]

    def get_vorarity(self, price_arr, cur_pos, period = None):
        tmp_arr = []
        prev = -1.0
        for val in price_arr[cur_pos-self.CHART_TYPE_JDG_LEN:cur_pos]:
            if prev == -1.0:
                tmp_arr.append(0.0)
            else:
                tmp_arr.append(val - prev)
            prev = val

        return np.std(tmp_arr)

    def get_macd(self, price_arr, cur_pos, period = 100):
        if cur_pos <= period:
            s = 0
        else:
            s = cur_pos - period
        tmp_arr = price_arr[s:cur_pos]
        tmp_arr.reverse()
        prices = np.array(tmp_arr, dtype=float)

        macd, macdsignal, macdhist = ta.MACD(prices,fastperiod=12, slowperiod=26, signalperiod=9)
        if macd[-1] > macdsignal[-1]:
            return 1
        else:
            return 0

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

    def make_serialized_data(self, start_idx, end_idx, step, x_arr_fpath, y_arr_fpath):
        input_mat = []
        angle_mat = []
        train_end_idx = -1
        print("all rate and data size: " + str(len(self.exchange_rates)))
        for i in range(start_idx, end_idx, step):
            if self.exchange_dates[i] == "2003-12-31 23:55:00":
                train_end_idx = i
            if i % 2000:
                print("current date idx: " + str(i))
            input_mat.append(
                [self.exchange_rates[i],
                 (self.exchange_rates[i] - self.exchange_rates[i - 1]) / self.exchange_rates[i - 1],
                 self.get_rsi(self.exchange_rates, i),
                 self.get_ma(self.exchange_rates, i),
                 self.get_ma_kairi(self.exchange_rates, i),
                 self.get_bb_1(self.exchange_rates, i),
                 self.get_bb_2(self.exchange_rates, i),
                 #self.get_ema(self.exchange_rates, i),
                 #self.get_cci(self.exchange_rates, i),
                 self.get_mo(self.exchange_rates, i),
                 self.get_vorarity(self.exchange_rates, i),
                 #self.get_macd(self.exchange_rates, i),
                 self.judge_chart_type(self.exchange_rates[i - self.CHART_TYPE_JDG_LEN:i])
                 ]
            )

            if y_arr_fpath != None:
                diff = self.exchange_rates[i + self.predict_future_legs] - self.exchange_rates[i]
                # if diff > self.half_spread:
                #     # BUY
                #     angle_mat.append([1.0, 0.0, 0.0])
                # elif diff < -1 * self.half_spread:
                #     # SELL
                #     angle_mat.append([0.0, 1.0, 0.0])
                # else:
                #     # DONOT
                #     angle_mat.append([0.0, 0.0, 1.0])

                # if diff > 0:
                #     # BUY
                #     angle_mat.append([1.0, 0.0])
                # else:
                #     # SELL
                #     angle_mat.append([0.0, 1.0])

                angle_mat.append([diff])

        input_mat = np.array(input_mat, dtype=np.float64)
        with open(x_arr_fpath, 'wb') as f:
            pickle.dump(input_mat, f)
        with open(y_arr_fpath, 'wb') as f:
            pickle.dump(angle_mat, f)
        print("test data end index: " + str(train_end_idx))

        return input_mat, angle_mat

    def setup_serialized_fx_data(self):
        self.exchange_dates = []
        self.exchange_rates = []

        if os.path.exists("./exchange_rates.pickle"):
            with open("./exchange_dates.pickle", 'rb') as f:
                self.exchange_dates = pickle.load(f)
            with open("./exchange_rates.pickle", 'rb') as f:
                self.exchange_rates = pickle.load(f)
        else:
            rates_fd = open('./USD_JPY_2001_2008_5min.csv', 'r')
            for line in rates_fd:
                splited = line.split(",")
                if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>" and splited[0] != "204/04/26" and splited[
                    0] != "20004/04/26" and self.is_weekend(splited[0]) == False:
                    time = splited[0].replace("/", "-")  # + " " + splited[1]
                    val = float(splited[1])
                    self.exchange_dates.append(time)
                    self.exchange_rates.append(val)
            # 足の長さを調節する
            self.exchange_dates = self.exchange_dates[::RATE_AND_DATE_STLIDE]
            self.exchange_rates = self.exchange_rates[::RATE_AND_DATE_STLIDE]
            with open("./exchange_rates.pickle", 'wb') as f:
                pickle.dump(self.exchange_rates, f)
            with open("./exchange_dates.pickle", 'wb') as f:
                pickle.dump(self.exchange_dates, f)

        if os.path.exists("./all_input_mat.pickle"):
            with open('./all_input_mat.pickle', 'rb') as f:
                all_input_mat = pickle.load(f)
            with open('./all_angle_mat.pickle', 'rb') as f:
                all_angle_mat = pickle.load(f)
        else:
            all_input_mat, all_angle_mat = \
                self.make_serialized_data(self.DATA_HEAD_ASOBI, len(self.exchange_rates) - self.DATA_HEAD_ASOBI - self.predict_future_legs, 1, './all_input_mat.pickle', './all_angle_mat.pickle')

        self.tr_input_arr, tr_scaler = self.preprocess_data(all_input_mat[0:self.COMPETITION_TRAIN_DATA_NUM])
        self.tr_angle_arr = all_angle_mat[0:self.COMPETITION_TRAIN_DATA_NUM]
        self.ts_input_arr, _ =  self.preprocess_data(all_input_mat[self.COMPETITION_TRAIN_DATA_NUM:self.COMPETITION_TRAIN_DATA_NUM * 2], tr_scaler)
        self.ts_angle_arr = all_angle_mat[self.COMPETITION_TRAIN_DATA_NUM:self.COMPETITION_TRAIN_DATA_NUM * 2]

        print("data size of all rates for train and test: " + str(len(self.exchange_rates)))
        print("num of rate datas for tarin: " + str(self.COMPETITION_TRAIN_DATA_NUM))
        print("input features sets for tarin: " + str(self.COMPETITION_TRAIN_DATA_NUM))
        print("input features sets for test: " + str(len(self.ts_input_arr)))
        print("finished setup environment data.")

    def get_env(self, type_str):
        if(type_str == "backtest"):
            return self.InnerFXEnvironment(self.tr_input_arr, self.exchange_dates, self.exchange_rates,
                                           self.DATA_HEAD_ASOBI, idx_step = 1, holdable_positions = self.holdable_positions, half_spred=self.half_spread,
                                           angle_arr=self.tr_angle_arr,  time_series = self.time_series, )

        elif(type_str == "backtest_test"):
            return self.InnerFXEnvironment(self.ts_input_arr, self.exchange_dates, self.exchange_rates,
                                           self.DATA_HEAD_ASOBI + self.COMPETITION_TRAIN_DATA_NUM, holdable_positions = self.holdable_positions, half_spread=self.half_spread,
                                           angle_arr=self.ts_angle_arr, time_series = self.time_series)
        else:
            raise Exception("unknown env_tye")

    def get_train_and_validation_datas(self):
        ret_tr_input_arr = []
        ret_ts_input_arr = []
        for idx in range(0, len(self.tr_input_arr) - self.time_series + 1):
            ret_tr_input_arr.append(self.tr_input_arr[idx:idx + self.time_series])
            ret_ts_input_arr.append(self.ts_input_arr[idx:idx + self.time_series])

        return ret_tr_input_arr, self.tr_angle_arr[:len(ret_tr_input_arr)], ret_ts_input_arr, self.ts_angle_arr[:len(ret_ts_input_arr)]

    class InnerFXEnvironment:
        def __init__(self, input_arr, exchange_dates, exchange_rates, idx_geta, time_series=32, angle_arr = None, half_spred=0.0015, holdable_positions=100, performance_eval_len = 20, reward_gamma = 0.95):
            self.BUY = 0
            self.SELL = 1
            self.DONOT = 2
            self.CLOSE = 3

            self.input_arr = input_arr
            self.angle_arr = angle_arr
            self.exchange_dates = exchange_dates
            self.exchange_rates = exchange_rates
            self.half_spread = half_spred
            self.time_series = time_series
            self.cur_idx = (time_series - 1) #LSTMで過去のstateを見るため、その分はずらしてスタートする
            self.idx_geta = idx_geta

            self.log_fd_bt = open("./backtest_log_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt", mode="w")

            self.start = time.time()
            self.idx_step = 1

            self.done = False
            self.holdable_positions = holdable_positions

            self.portfolio_mngr = PortforioManager(exchange_rates, self.half_spread, holdable_position_num = self.holdable_positions)

        def get_rand_str(self):
            return str(random.randint(0, 10000000))

        def logfile_writeln_bt(self, log_str):
            self.log_fd_bt.write(log_str + "\n")
            self.log_fd_bt.flush()

        def close_all(self, cur_episode_rate_idx):
            won_pips, won_money, each_pos_won = self.portfolio_mngr.close_all(cur_episode_rate_idx)

            for idx in range(0, len(self.positions_identifiers)):
                # 対象のエピソードが対応する  all_period_reward_arr のインデックスに対応させる
                # 最小の値は time_series - 1 で良い
                episode_idx_of_past_open = each_pos_won[idx][1] - self.idx_geta
                # エピソードの識別子,そのエピソードでのポジションのオープンによる獲得pips,ポジションをオープンした時のイテレーション上のインデックス,ポジションの種類
                self.additional_infos.append([self.positions_identifiers[idx], each_pos_won[idx][0], episode_idx_of_past_open, each_pos_won[idx][2]])

            self.positions_identifiers = []

            return won_pips, won_money

        def step(self, action_num):
            cur_step_identifier = self.get_rand_str()
            cur_episode_rate_idx = self.idx_geta + self.cur_idx

            if action_num == self.BUY:
                action = "BUY"
            elif action_num == self.SELL:
                action = "SELL"
            elif action_num == self.DONOT:
                 action = "DONOT"
            elif action_num == self.CLOSE:
                 action = "CLOSE"
            else:
                raise Exception(str(action_num) + " is invalid.")

            a_log_str_line = "log," + str(self.cur_idx) + "," + action
            self.additional_infos = []

            if action == "BUY":
                if self.portfolio_mngr.additional_pos_openable():
                    buy_val = self.portfolio_mngr.buy(cur_episode_rate_idx)
                    self.positions_identifiers.append(cur_step_identifier)
                    a_log_str_line += ",OPEN_LONG" + ",0,0," + str(
                    self.exchange_rates[cur_episode_rate_idx]) + "," + str(buy_val)
                else: #もうオープンできない
                    a_log_str_line += ",POSITION_HOLD,0," + str(self.portfolio_mngr.get_evaluated_val_diff_of_all_pos(cur_episode_rate_idx)) + "," + str(
                    self.exchange_rates[cur_episode_rate_idx]) + ",0"
            if action == "SELL":
                if self.portfolio_mngr.additional_pos_openable():
                    buy_val = self.portfolio_mngr.sell(cur_episode_rate_idx)
                    self.positions_identifiers.append(cur_step_identifier)
                    a_log_str_line += ",OPEN_SHORT" + ",0,0," + str(
                    self.exchange_rates[cur_episode_rate_idx]) + "," + str(buy_val)
                else: #もうオープンできない
                    a_log_str_line += ",POSITION_HOLD,0," + str(self.portfolio_mngr.get_evaluated_val_diff_of_all_pos(cur_episode_rate_idx)) + "," + str(
                    self.exchange_rates[cur_episode_rate_idx]) + ",0"
            elif action == "CLOSE":
                if len(self.positions_identifiers) > 0:
                    won_pips, won_money = self.close_all(cur_episode_rate_idx)
                    a_log_str_line += ",CLOSE_LONG_AND_DONOT" + "," + str(won_money) + "," + str(
                       won_pips) + "," + str(self.exchange_rates[cur_episode_rate_idx]) + ",0"
                else:
                    a_log_str_line += ",KEEP_NO_POSITION" + ",0,0," + str(self.exchange_rates[cur_episode_rate_idx]) + ",0"
            elif action == "DONOT":
                if len(self.positions_identifiers) > 0:
                    operation_str = ",POSITION_HOLD,0,"

                    a_log_str_line += operation_str + "0," + str(
                        self.exchange_rates[cur_episode_rate_idx]) + ",0"
                else:
                    a_log_str_line += ",KEEP_NO_POSITION,0,0," + str(
                        self.exchange_rates[cur_episode_rate_idx]) + ",0"

            else:
                raise Exception(str(action) + " is invalid.")

            a_log_str_line += "," + str(self.portfolio_mngr.get_current_portfolio(cur_episode_rate_idx)) +\
                              "," + str(self.portfolio_mngr.total_won_pips) + "," + str(self.portfolio_mngr.having_money) + "," + str(self.portfolio_mngr.get_nomal_position_num())


            self.logfile_writeln_bt(a_log_str_line)

            self.cur_idx += self.idx_real_step #self.idx_step
            if (self.cur_idx) >= (len(self.input_arr) - (self.time_series - 1) - 1):
                self.logfile_writeln_bt("finished backtest.")
                print("finished backtest.")
                process_time = time.time() - self.start
                self.logfile_writeln_bt("excecution time of backtest: " + str(process_time))
                self.logfile_writeln_bt("result of portfolio: " + str(self.portfolio_mngr.get_current_portfolio(cur_episode_rate_idx)))
                print("result of portfolio: " + str(self.portfolio_mngr.get_current_portfolio(cur_episode_rate_idx)))
                self.log_fd_bt.flush()
                self.log_fd_bt.close()
                return None, True
            else:
                next_state = self.input_arr[self.cur_idx - self.time_series + 1:self.cur_idx + 1]
                # 第四返り値はエピソードの識別子を格納するリスト. 第0要素は返却する要素に対応するもので、
                # それ以外の要素がある場合は、close時にさかのぼって エピソードのrewardを更新するためのもの
                return next_state, False

class PortforioManager:

    def __init__(self, exchange_rates, half_spred=0.0015, holdable_position_num = 100, is_backtest=False):
        self.holdable_position_num = holdable_position_num
        self.exchange_rates = exchange_rates
        self.half_spread = half_spred
        self.is_backtest = is_backtest

        self.LONG = 0
        self.SHORT = 1
        self.NOT_HAVE = 2 #CLOSE

        self.having_money = 1000000.0
        self.total_won_pips = 0.0

        # 各要素は [購入時の価格（スプレッド含む）, self.LONG or self.SHORT, 数量]
        # 数量は通貨の数を表しLONGであれば正、SHORTであれば負の値となる
        self.positions = []
        self.position_num = 0

        # # NOT_HAVE
        # self.having_long_or_short = self.NOT_HAVE

    def get_nomal_position_num(self):
        return self.position_num

    # ダミーのポジションを含まずにチェックされる
    def additional_pos_openable(self):
        return self.position_num < self.holdable_position_num

    # 規約: 保持可能なポジション数を超える場合は呼び出されない
    # ロングポジションを最大保持可能数における1単位分購入する（購入通貨数が整数になるような調整は行わない）
    def buy(self, rate_idx):
        pos_kind = self.LONG
        trade_val = self.exchange_rates[rate_idx] + self.half_spread
        currency_num = (self.having_money / (self.holdable_position_num - self.position_num)) / trade_val

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
        currency_num = (self.having_money / (self.holdable_position_num - self.position_num)) / trade_val

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
        won_pips_arr = []
        for position in self.positions:
            if position[1] == self.LONG:
                won_pips, won_money, return_money = self.close_long(position, rate_idx)
            elif position[1] == self.SHORT: # self.SHORT
                won_pips, won_money, return_money = self.close_short(position, rate_idx)

            # 獲得pips, エピソードの1イテレーションの中でのインデックス（rateのidxである点に注意）, ポジションの種類
            won_pips_arr.append([won_pips, position[3], position[1]])
            won_pips_sum += won_pips
            won_money_sum += won_money
            returned_money_sum += return_money

        #self.having_long_or_short = self.NOT_HAVE
        self.total_won_pips += won_pips_sum
        self.having_money += returned_money_sum
        self.positions = []
        self.position_num = 0

        return won_pips_sum, won_money_sum, won_pips_arr

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

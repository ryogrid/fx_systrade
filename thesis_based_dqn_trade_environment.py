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
    def __init__(self, train_data_num, time_series=32, holdable_positions=100, half_spread=0.0015):
        print("FXEnvironment class constructor called.")
        self.COMPETITION_TRAIN_DATA_NUM = train_data_num

        self.DATA_HEAD_ASOBI = 65

        self.tr_input_arr = None
        self.val_input_arr = None

        self.exchange_dates = None
        self.exchange_rates = None

        self.time_series = time_series
        self.holdable_positions  = holdable_positions

        self.setup_serialized_fx_data()
        self.half_spread = half_spread

    def preprocess_data(self, X, scaler=None):
        if scaler == None:
            scaler = StandardScaler()
            scaler.fit(X)

        X_T = scaler.transform(X)
        return X_T, scaler

    def get_rsi(self, price_arr, cur_pos, period = 30):
        if cur_pos <= period:
            return 0
        else:
            s = cur_pos - (period + 1)
        tmp_arr = price_arr[s:cur_pos]
        #tmp_arr.reverse()
        prices = np.array(tmp_arr, dtype=float)

        return ta.RSI(prices, timeperiod = period)[-1]

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

    #TODO: need EMSD version
    def get_vorarity(self, price_arr, cur_pos, period = None):
        tmp_arr = []
        prev = -1.0
        for val in price_arr[cur_pos-period:cur_pos]:
            if prev == -1.0:
                tmp_arr.append(0.0)
            else:
                tmp_arr.append(val - prev)
            prev = val

        return np.std(tmp_arr)

    def get_macd(self, price_arr, cur_pos, period = 63):
        if cur_pos <= period:
            s = 0
        else:
            s = cur_pos - period
        tmp_arr = price_arr[s:cur_pos]
        #tmp_arr.reverse()
        prices = np.array(tmp_arr, dtype=float)

        macd, macdsignal, macdhist = ta.MACD(prices, fastperiod=8, slowperiod=24)

        return macd[-1]

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
        angle_mat = []
        train_end_idx = -1
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

        return input_mat, angle_mat

    def setup_serialized_fx_data(self):
        self.exchange_dates = []
        self.exchange_rates = []

        if False: #self.is_fist_call == False and os.path.exists("./exchange_rates.pickle"):
            with open("./exchange_dates.pickle", 'rb') as f:
                self.exchange_dates = pickle.load(f)
            with open("./exchange_rates.pickle", 'rb') as f:
                self.exchange_rates = pickle.load(f)
        else:
            rates_fd = open('./USD_JPY_2001_2008_5min.csv', 'r')
            # 日足のデータを得る
            for line in rates_fd:
                splited = line.split(",")
                if "23:55:00" in splited[0] and splited[2] != "High" and splited[0] != "<DTYYYYMMDD>" and splited[0] != "204/04/26" and splited[
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

        if False: #self.is_fist_call == False and os.path.exists("./all_input_mat.pickle"):
            with open('./all_input_mat.pickle', 'rb') as f:
                all_input_mat = pickle.load(f)
        else:
            all_input_mat = \
                self.make_serialized_data(self.DATA_HEAD_ASOBI, len(self.exchange_rates) - self.DATA_HEAD_ASOBI, 1, './all_input_mat.pickle')

        # TODO: 論文では価格のみ normalize したとあるが、面倒なので全ての特徴量を normalize してしまう
        self.tr_input_arr, tr_scaler = self.preprocess_data(all_input_mat[0:self.COMPETITION_TRAIN_DATA_NUM])
        self.ts_input_arr, _ =  self.preprocess_data(all_input_mat[self.COMPETITION_TRAIN_DATA_NUM:], tr_scaler)

        print("data size of all rates for train and test: " + str(len(self.exchange_rates)))
        print("input features sets for tarin: " + str(self.COMPETITION_TRAIN_DATA_NUM))
        print("input features sets for test: " + str(len(self.ts_input_arr)))
        print("finished setup environment data.")

    # type_str: "train", "test"
    def get_env(self, type_str):
        if(type_str == "backtest"):
            return self.InnerFXEnvironment(self.tr_input_arr, self.exchange_dates, self.exchange_rates,
                                           self.DATA_HEAD_ASOBI, holdable_positions = self.holdable_positions, half_spread=self.half_spread,
                                           time_series = self.time_series, is_backtest=True)
        if(type_str == "auto_backtest"):
            return self.InnerFXEnvironment(self.tr_input_arr, self.exchange_dates, self.exchange_rates,
                                           self.DATA_HEAD_ASOBI, holdable_positions = self.holdable_positions, half_spread=self.half_spread,
                                           time_series = self.time_series, is_backtest=True, is_auto_backtest=True)
        elif(type_str == "backtest_test"):
            return self.InnerFXEnvironment(self.ts_input_arr, self.exchange_dates, self.exchange_rates,
                                           self.DATA_HEAD_ASOBI + self.COMPETITION_TRAIN_DATA_NUM, holdable_positions = self.holdable_positions, half_spread=self.half_spread,
                                           time_series = self.time_series, is_backtest=True)
        elif(type_str == "auto_backtest_test"):
            return self.InnerFXEnvironment(self.ts_input_arr, self.exchange_dates, self.exchange_rates,
                                           self.DATA_HEAD_ASOBI + self.COMPETITION_TRAIN_DATA_NUM, holdable_positions = self.holdable_positions, half_spread=self.half_spread,
                                           time_series = self.time_series, is_backtest=True, is_auto_backtest = True)
        else:
            return self.InnerFXEnvironment(self.tr_input_arr, self.exchange_dates, self.exchange_rates,
                                           self.DATA_HEAD_ASOBI, time_series = self.time_series, holdable_positions = self.holdable_positions, half_spread=self.half_spread,
                                           is_backtest=False)

    class InnerFXEnvironment:
        def __init__(self, input_arr, exchange_dates, exchange_rates, idx_geta, time_series=32, half_spread=0.0015, holdable_positions=100, is_backtest = False, is_auto_backtest = False):
            self.LONG = 0
            self.SHORT = 1
            self.NOT_HAVE = 2

            self.input_arr = input_arr
            self.exchange_dates = exchange_dates
            self.exchange_rates = exchange_rates
            self.half_spread = half_spread
            self.time_series = time_series
            self.cur_idx = (time_series - 1) #LSTMで過去のstateを見るため、その分はずらしてスタートする
            self.idx_geta = idx_geta
            self.is_backtest = is_backtest
            self.is_auto_backtest = is_auto_backtest

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
            self.input_arr_len = len(input_arr)
            self.holdable_positions = holdable_positions
            # if(is_backtest == False):
            #     self.half_spread = 0.0

            self.portfolio_mngr = PortforioManager(exchange_rates, self.half_spread, holdable_position_num = self.holdable_positions)

        def get_rand_str(self):
            return str(random.randint(0, 10000000))

        def logfile_writeln_bt(self, log_str):
            self.log_fd_bt.write(log_str + "\n")
            self.log_fd_bt.flush()

        def close_all(self, cur_episode_rate_idx):
            won_pips, won_money = self.portfolio_mngr.close_all(cur_episode_rate_idx)

            return won_pips, won_money

        def step(self, action_num):
            cur_episode_rate_idx = self.idx_geta + self.cur_idx
            did_close = False

            if action_num == -1:
                action = "SELL"
            elif action_num == 0:
                action = "DONOT"
            elif action_num == 1:
                 action = "BUY"
            else:
                raise Exception(str(action_num) + " is invalid.")

            a_log_str_line = "log," + str(self.cur_idx) + "," + action

            if action == "BUY":
                reward = 1

                # TODO: rewardの計算が必要

                if self.portfolio_mngr.having_long_or_short == self.NOT_HAVE:
                    buy_val = self.portfolio_mngr.buy(cur_episode_rate_idx)
                    a_log_str_line += ",OPEN_LONG" + ",0,0," + str(
                    self.exchange_rates[cur_episode_rate_idx]) + "," + str(buy_val)
                elif self.portfolio_mngr.having_long_or_short == self.SHORT:
                    #持っているSHORTポジションをクローズ
                    self.close_all(cur_episode_rate_idx)
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
                reward = 0

                # TODO: rewardの計算が必要

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
                reward = -1

                # TODO: rewardの計算が必要

                if self.portfolio_mngr.having_long_or_short == self.NOT_HAVE:
                    sell_val = self.portfolio_mngr.sell(cur_episode_rate_idx)
                    a_log_str_line += ",OPEN_SHORT" + ",0,0," + str(
                    self.exchange_rates[cur_episode_rate_idx]) + "," + str(sell_val)
                elif self.portfolio_mngr.having_long_or_short == self.LONG:
                    #持っているLONGポジションをクローズ
                    self.close_all(cur_episode_rate_idx)
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

            self.cur_idx += self.idx_step
            if (self.cur_idx) >= (len(self.input_arr) - (self.time_series - 1) - 1):
                self.logfile_writeln_bt("finished backtest.")
                print("finished backtest.")
                process_time = time.time() - self.start
                self.logfile_writeln_bt("excecution time of backtest: " + str(process_time))
                self.logfile_writeln_bt("result of portfolio: " + str(self.portfolio_mngr.get_current_portfolio(cur_episode_rate_idx)))
                print("result of portfolio: " + str(self.portfolio_mngr.get_current_portfolio(cur_episode_rate_idx)))
                self.log_fd_bt.flush()
                self.log_fd_bt.close()
                return None, reward, True

                next_state = self.input_arr[self.cur_idx - self.time_series + 1:self.cur_idx + 1]
                # 第四返り値はエピソードの識別子を格納するリスト. 第0要素は返却する要素に対応するもので、
                # それ以外の要素がある場合は、close時にさかのぼって エピソードのrewardを更新するためのもの
                return next_state, reward, False

class PortforioManager:

    def __init__(self, exchange_rates, half_spred=0.0015, holdable_position_num = 100, is_backtest=False):
        self.holdable_position_num = holdable_position_num
        self.exchange_rates = exchange_rates
        self.half_spread = half_spred
        self.is_backtest = is_backtest

        self.LONG = 0
        self.SHORT = 1
        self.NOT_HAVE = 2

        self.having_money = 1000000.0
        self.fixed_use_money_mue = 500000.0
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
        currency_num = (self.fixed_use_money_mue / (self.holdable_position_num - self.position_num)) / trade_val

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
        currency_num = (self.fixed_use_money_mue / (self.holdable_position_num - self.position_num)) / trade_val

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

# coding:utf-8
# [0]必要なライブラリのインポート

# this code based on code on https://qiita.com/sugulu/items/bc7c70e6658f204f85f9
# I am very grateful to work of Mr. Yutaro Ogawa (id: sugulu)

import numpy as np
from keras.models import Sequential, model_from_json, Model, load_model
from keras.layers import Dense, BatchNormalization, Dropout, LSTM, RepeatVector, TimeDistributed, Reshape, LeakyReLU
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from collections import deque
from keras import backend as K
import tensorflow as tf
import pickle
from agent_fx_environment_lstm import FXEnvironment
import os
import sys
import random
import itertools

# [1]損失関数の定義
# 損失関数にhuber関数を使用 参考https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
   err = y_true - y_pred
   cond = K.abs(err) < 1.0
   L2 = 0.5 * K.square(err)
   L1 = (K.abs(err) - 0.5)
   loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
   return K.mean(loss)

# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.001, state_size=15, action_size=3, time_series=32):
        global all_period_reward_arr

        self.model = Sequential()

        # self.model.add(LSTM(time_series, activation='relu', kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001),
        #                     bias_regularizer=l2(0.001), input_shape=(state_size, time_series), return_sequences=True))
        # self.model.add(LSTM(time_series, activation='relu', kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001),
        #                     bias_regularizer=l2(0.001), return_sequences=False))
        # self.model.add(Dense(time_series, activation='linear', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
        # self.model.add(Dense(action_size, activation='linear',kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))

        #self.model.add(LSTM(time_series, activation='relu', input_shape=(state_size, time_series), return_sequences=True))

        # self.model.add(
        #     LSTM(hidden_size, activation='relu', input_shape=(time_series, state_size), return_sequences=True))
        # self.model.add(LSTM(hidden_size, activation='relu', return_sequences=False))
        self.model.add(
            LSTM(hidden_size, input_shape=(time_series, state_size), return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU(0.2))
        self.model.add(LSTM(hidden_size, return_sequences=False))
        self.model.add(LeakyReLU(0.2))

        # self.model.add(Dense(hidden_size, activation='linear'))
        # self.model.add(Dense(hidden_size))
        # self.model.add(LeakyReLU(0.2))

        self.model.add(Dense(action_size, activation='linear'))

        self.optimizer = Adam(lr=learning_rate, clipvalue=5.0)
        #self.optimizer = SGD(lr=learning_rate, momentum=0.9, clipvalue=5.0)

        self.model.compile(optimizer=self.optimizer, loss=huberloss)
        #self.model.compile(optimizer=self.optimizer, loss="mae")

        self.model.summary()

        self.buy_donot_diff_memory_predicted = Memory([], all_period_reward_arr = all_period_reward_arr, max_size=10000) # predictしたBUYとDONOTの報酬の絶対値の差分を保持する
        self.buy_donot_diff_memory_collect = Memory([], all_period_reward_arr = all_period_reward_arr, max_size=TRAIN_DATA_NUM) # 配列で保持しているBUYとDONOTの報酬の平均値の絶対値の差分を保持する

        self.batch_num_to_call_fit = 80
        self.batch_datas_for_generator = []

    # 重みの学習
    def replay(self, memory, time_series, cur_episode_idx = 0):
        inputs = np.zeros((batch_size, time_series, feature_num))
        targets = np.zeros((batch_size, 1, nn_output_size))

        mini_batch = memory.get_sequencial_samples(batch_size, (cur_episode_idx + 1) - batch_size)
        # rewardだけ別管理の平均値のリストに置き換える
        mini_batch = memory.get_sequencial_converted_samples(mini_batch, (cur_episode_idx + 1) - batch_size)

        for idx, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            # state_b_T = state_b.T #.copy()
            # reshaped_state = np.reshape(state_b_T, [1, time_series, feature_num])
            reshaped_state = np.reshape(state_b, [1, time_series, feature_num])
            inputs[idx] = reshaped_state
            targets[idx] = np.reshape(self.model.predict(reshaped_state)[0], [1, nn_output_size])

            # 学習の進行度の目安としてBUYとDONOTの predict した報酬の絶対値の差の平均値を求める
            # （理想的に学習していれば、両者は絶対値が同じ符号が逆の値になるはずであるので、0に近づくほど学習が進んでいると見なせる、はず）
            # residual は 残差 の意
            # 今のパラメータでは1万差分の平均をスライドさせながら基本的に出力する.
            # 1万にデータが満たない場合は、その範囲での平均を出力する
            self.buy_donot_diff_memory_predicted.add(abs(abs(targets[idx][0][0]) - abs(targets[idx][0][2])))
            agent_learn_residual = self.buy_donot_diff_memory_predicted.get_mean_value()
            # 正解の方も求めて出力
            self.buy_donot_diff_memory_collect.add_buy_donot_abs_diff(cur_episode_idx + idx)
            base_data_residual = self.buy_donot_diff_memory_collect.get_mean_value()

            # print(targets.shape)
            # sys.exit(1)
            predicted_buy = targets[idx][0][0]
            predicted_donot = targets[idx][0][2]
            
            buy_donot_diff = predicted_buy - predicted_donot
            # 符号が同じかどうか判定するための下処理
            buy_donot_diff_abs = abs(buy_donot_diff)
            buy_donot_diff_abs_abs = abs(abs(predicted_buy) - abs(predicted_donot))
            
            print("reward_b: BUY -> " + str(predicted_buy) + "," + str(reward_b[0]) +
                  " CLOSE -> " + str(targets[idx][0][1]) +
                  " DONOT -> " + str(predicted_donot) + "," + str(reward_b[2]) +
                  " (BUY - DONOT) -> " + str(buy_donot_diff) + "," + str(True if buy_donot_diff_abs > buy_donot_diff_abs_abs else False) +
                  " predicted residual -> " + str(agent_learn_residual) +
                  " collect residual -> " + str(base_data_residual)
            )

            # イテレーションをまたいで平均rewardを計算しているlistから3つ全てのアクションのrewardを得てあるので
            # 全て設定する
            
            # targets[idx][0][0] = reward_b[0] # 教師信号
            # targets[idx][0][1] = -100.0      # CLOSEのrewardは必ず-100.0
            # targets[idx][0][2] = reward_b[2] # 教師信号

            # BUYとDONOTの教師信号は符号で-1, 1 にクリッピングする
            targets[idx][0][0] = 1.0 if reward_b[0] > 0 else -1.0 # 教師信号
            targets[idx][0][1] = -100.0  # CLOSEのrewardは必ず-100.0
            targets[idx][0][2] = 1.0 if reward_b[2] > 0 else -1.0  # 教師信号

        targets = np.array(targets)
        inputs = np.array(inputs)

        inputs = inputs.reshape((batch_size, time_series, feature_num))
        targets = targets.reshape((batch_size, nn_output_size))

        self.model.fit(inputs, targets, epochs=1, verbose=1, batch_size=batch_size)

        # # 複数コアで並列に処理するため、複数バッチが貯まったら git_generatorで fit を行う
        # # Windows環境では動作しない
        # def batch_generator():
        #     nonlocal self
        #     for idx, (x, y) in enumerate(self.batch_datas_for_generator):
        #         yield (x, y)
        #
        # self.batch_datas_for_generator.append([inputs, targets])
        # if len(self.batch_datas_for_generator) >= self.batch_num_to_call_fit:
        #     self.model.fit_generator(batch_generator(), epochs=1, shuffle=False, workers=8, use_multiprocessing=True, verbose=1, steps_per_epoch=self.batch_num_to_call_fit)
        #     self.batch_datas_for_generator = []

    def save_model(self, file_path_prefix_str):
        self.model.save("./" + file_path_prefix_str + ".hd5")
        # with open("./" + file_path_prefix_str + "_nw.json", "w") as f:
        #     f.write(self.model.to_json())
        # self.model.save_weights("./" + file_path_prefix_str + "_weights.hd5")

    def load_model(self, file_path_prefix_str):
        self.model = load_model("./" + file_path_prefix_str + ".hd5", compile=False)
        # with open("./" + file_path_prefix_str + "_nw.json", "r") as f:
        #     self.model = model_from_json(f.read())
        # self.model.compile(loss=huberloss, optimizer=self.optimizer)
        # self.model.load_weights("./" + file_path_prefix_str + "_weights.hd5")

# [3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, initial_elements, max_size=1000, all_period_reward_arr=None):
        self.max_size = max_size
        self.buffer = deque(initial_elements, maxlen=max_size)
        if all_period_reward_arr != None:
            self.all_period_reward_arr = all_period_reward_arr

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def get_last(self, num):
        deque_length = len(self.buffer)
        start = deque_length - num
        end = deque_length
        return [self.buffer[ii] for ii in range(start, end)]

    # 呼び出し側がmemory内の適切なstart要素インデックスを計算して呼び出す
    def get_sequencial_samples(self, batch_size, start_idx):
        print(start_idx)
        return [self.buffer[ii] for ii in range(start_idx, start_idx + batch_size)]

    # 連続したエピソードのサンプルシストを渡して、イテレーションにおける開始インデックス
    # を指定すると、reward情報をよろしく入れ替えて返す
    # 具体的には保持している全イテレーション共通の各episode x action のリストの情報を利用し、
    # 以下の形状のデータを返す. 置き換えているのはrewardだけだが3要素のリストにする
    # [state, action, reward(3要素のリスト), next_episode, info]
    def get_sequencial_converted_samples(self, base_data, start_idx):
        ret_list = []
        print(start_idx)
        for idx, (state, action, reward, next_action) in enumerate(base_data):
            ret_list.append([state, action, self.all_period_reward_arr[start_idx + idx], next_action])

        return ret_list

    #第2引数で指定した要素数分の末尾の要素の中から、第一引数で指定した数
    # だけの連続したepisodeを返す
    def get_random_sequencial_samples(self, batch_size, last_len):
        deque_length = len(self.buffer)
        ok_part_start_idx = deque_length - last_len - 1
        # シーケンスの先頭として指定できるインデックスの最大値
        ok_part_end_idx = deque_length - batch_size - 1
        seq_start = random.randint(ok_part_start_idx, ok_part_end_idx)
        seq_end = seq_start + batch_size
        return [self.buffer[ii] for ii in range(seq_start, seq_end)]

    def add_buy_donot_abs_diff(self, episode_idx):
        self.add(abs(abs(self.all_period_reward_arr[episode_idx][0]) - abs(self.all_period_reward_arr[episode_idx][2])))

    def sum(self):
        return sum(list(self.buffer))

    def get_mean_value(self):
        return self.sum() / self.len()

    def len(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = deque(maxlen=self.max_size)

    def save_memory(self, file_path_prefix_str):
        with open("./" + file_path_prefix_str + ".pickle", 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_memory(self, file_path_prefix_str):
        with open("./" + file_path_prefix_str + ".pickle", 'rb') as f:
            self.buffer = pickle.load(f)

# [4]カートの状態に応じて、行動を決定するクラス
class Actor:
    def get_action(self, state, experienced_episodes, mainQN, cur_itr, isBacktest = False):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0 + (300.0 * (experienced_episodes / TOTAL_ACTION_NUM)))

        # epsilonが小さい値の場合の方が最大報酬の行動が起こる
        # 周回数が3の倍数の時か、バックテストの場合は常に最大報酬の行動を選ぶ
        if epsilon <= np.random.uniform(0, 1) or isBacktest == True or ((cur_itr % 5 == 0) and cur_itr != 0):
            # バッチサイズ個の予測結果が返ってくるので最後の1アウトプットのみ見る
            # state_T = state.T #.copy()
            # reshaped_state = np.reshape(state_T, [1, time_series, feature_num])
            reshaped_state = np.reshape(state, [1, time_series, feature_num])
            retTargetQs = mainQN.model.predict(reshaped_state)
            print("NN all output at get_action: " + str(list(itertools.chain.from_iterable(retTargetQs))))
            #print("NN output [0] at get_action: " + str(list(itertools.chain.from_iterable(retTargetQs[0]))))
            #print(list(itertools.chain.from_iterable(retTargetQs[-1])))
            # 1要素しかないが、複数返ってくるように修正した場合を想定して -1 を指定
            #action = np.argmax(retTargetQs[-1])  # 最大の報酬を返す行動を選択する
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
        else:
            action = np.random.choice([0, 1, 2])  # ランダムに行動する

        return action

# [5] メイン関数開始----------------------------------------------------
# [5.1] 初期設定--------------------------------------------------------

# ---
#gamma = 0.95 # <- 今の実装では利用されていない #0.99 #0.3 # #0.99 #0.3 #0.99  # 割引係数
hidden_size = 64 #32 #24 #50 #28 #80 #28 #50 # <- 50層だとバッチサイズ=32のepoch=1で1エピソード約3時間かかっていた # Q-networkの隠れ層のニューロンの数
learning_rate = 0.0001 #0.01 #0.001 #0.01 #0.0005 # 0.0005 #0.0001 #0.005 #0.01 # 0.05 #0.001 #0.0001 # 0.00001         # Q-networkの学習係数
time_series = 64 #32 #64 #32
batch_size = 64 #8 #64 #8 #1 #64 #16 #32 #16 #32 #64 # 32  # Q-networkを更新するバッチの大きさ
TRAIN_DATA_NUM = 36000 - time_series #テストデータでうまくいくまで半年に減らす  #74651 # <- 検証中は期間を1年程度に減らす　223954 # 3years (test is 5 years)
num_episodes = TRAIN_DATA_NUM + 10  # envがdoneを返すはずなので念のため多めに設定 #1000  # 総試行回数
iteration_num = 720 # <- 劇的に減らす(1足あたり 16 * 1 * 50 で800回のfitが行われる計算) #720 #20
#memory_size = TRAIN_DATA_NUM * iteration_num + 10 #TRAIN_DATA_NUM * int(iteration_num * 0.2) # 全体の20%は収まるサイズ. つまり終盤は最新の当該割合に対応するエピソードのみreplayする #10000
memory_size = TRAIN_DATA_NUM * 2 + 10 #TRAIN_DATA_NUM * int(iteration_num * 0.2) # 全体の20%は収まるサイズ. つまり終盤は最新の当該割合に対応するエピソードのみreplayする #10000
feature_num = 10 #10 + 1 #10 + 9*3 #10 #11 #10 #11 #10 #11
nn_output_size = 3
TOTAL_ACTION_NUM = TRAIN_DATA_NUM * iteration_num
holdable_positions = 100 #30 # 100

# イテレーションを跨いで、ある足での action に対する reward の平均値を求める際に持ちいる時間割引率
# 昔に得られた結果だからといって割引してはCLOSEのタイミングごとに平等に反映されないことになるので
# 現在の実装では 1.0 とする
gamma_at_reward_mean = 1.0 #0.9

NOT_HAVE = 0
LONG = 1
SHORT = 2

all_period_reward_arr = [[0.0, -100.0, 0.0] for i in range(TRAIN_DATA_NUM)]

def tarin_agent():
    global all_period_reward_arr

    env_master = FXEnvironment(time_series=time_series, holdable_positions=holdable_positions)

    # [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    mainQN = QNetwork(time_series=time_series, learning_rate=learning_rate, state_size=feature_num, action_size=nn_output_size)     # メインのQネットワーク

    memory = Memory([], max_size=memory_size, all_period_reward_arr=all_period_reward_arr)
    memory_hash = {}
    actor = Actor()

    total_get_acton_cnt = 1
    all_period_reward_hash = {}
    # イテレーションを跨いで state x action で最新のエピソードのみ記録して、replay可能とするため
    # のハッシュ. 必要なのはrewardの値だが実装の都合上不要な情報も含む
    # 値は [state, action, reward, next_state]
    state_x_action_hash = {}

    # if os.path.exists("./mainQN_nw.json"):
    if os.path.exists("./mainQN.hd5"):
        mainQN.load_model("mainQN")
        memory.load_memory("memory")
        with open("./total_get_action_count.pickle", 'rb') as f:
            total_get_acton_cnt = pickle.load(f)
        with open("./all_period_reward_hash.pickle", 'rb') as f:
            all_period_reward_hash = pickle.load(f)
        with open("./state_x_action_hash.pickle", 'rb') as f:
            state_x_action_hash = pickle.load(f)
        with open("./all_period_reward_arr.pickle", 'rb') as f:
            all_period_reward_arr = pickle.load(f)

    def store_episode_log_to_memory(state, action, reward, next_state, info):
        nonlocal memory
        nonlocal memory_hash
        nonlocal state_x_action_hash
        a_log = [state, action, reward, next_state]
        memory.add(a_log)  # メモリを更新する
        # 後からrewardを更新するためにエピソード識別子をキーにエピソードを取得可能としておく
        memory_hash[info[0]] = a_log
        key_str = str(state) + str(action)
        # 最新の値で更新する、もしくはエントリが無い場合は追加する
        # ハッシュが持っている要素はエピソードの記録（リストで実装されている）への参照であるため、
        # 当該エピソードの記録が更新された場合、state_x_action_hashが保持するデータも更新
        # されることになる

        # 本来不要な情報だが、memoryの中のオブジェクトが更新された場合に
        # 情報が更新されるよう、episodeの情報全てへの参照を持つ形にする
        # 既にエントリがあろうが無かろうか最新のepisodeへの参照を設定してしまう
        state_x_action_hash[key_str] = a_log

    #######################################################

    for cur_itr in range(iteration_num):
        env = env_master.get_env('train')
        action = np.random.choice([0, 1, 2])
        state, reward, done, info, needclose = env.step(action)  # 1step目は適当な行動をとる
        total_get_acton_cnt += 1
        state = np.reshape(state, [time_series, feature_num])  # list型のstateを、1行15列の行列に変換
        # ここだけ 同じstateから同じstateに遷移したことにする
        store_episode_log_to_memory(state, action, reward, state, info)

        # スナップショットをとっておく
        if cur_itr % 5 == 0 and cur_itr != 0:
            # targetQN.save_model("targetQN")
            mainQN.save_model("mainQN")
            memory.save_memory("memory")
            with open("./total_get_action_count.pickle", 'wb') as f:
                pickle.dump(total_get_acton_cnt, f)
            with open("./all_period_reward_hash.pickle", 'wb') as f:
                pickle.dump(all_period_reward_hash, f)
            with open("./state_x_action_hash.pickle", 'wb') as f:
                pickle.dump(state_x_action_hash, f)
            with open("./all_period_reward_arr.pickle", 'wb') as f:
                pickle.dump(all_period_reward_arr, f)

        for episode in range(num_episodes):  # 試行数分繰り返す
            total_get_acton_cnt += 1
            if needclose:
                action = 1
            else:
                action = actor.get_action(state, total_get_acton_cnt, mainQN, cur_itr)  # 時刻tでの行動を決定する
            next_state, reward, done, info, needclose = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
            # 環境が提供する期間が最後までいった場合
            if done:
                print(str(cur_itr) + ' training period finished.')
                # next_stateは今は使っていないのでreshape等は不要
                # total_get_actionと memory 内の要素数がズレるのを避けるために追加しておく
                store_episode_log_to_memory(state, action, reward, next_state, info)
                break
            next_state = np.reshape(next_state, [time_series, feature_num])  # list型のstateを、1行feature num列の行列に変換

            store_episode_log_to_memory(state, action, reward, next_state, info)

            # closeされた場合過去のBUY, DONOTについて獲得pipsに係数をかけた値が与えられる.
            # 各Actionについての獲得pipsが識別子文字列とともにinfo で返されるので、過去のイテレーションでの平均値を踏まえて、
            # 今回のイテレーションでのリワードを更新し、過去のイテレーションでの平均値も更新する
            if len(info) > 1:
                for keyval in info[1:]:
                    # rewardは過去の値の寄与度も考慮した平均値になるように設定する
                    current_val = -1
                    # 同じ足についてstateは各イテレーションで共通なので、 state と action を文字列として結合したものをキーとして
                    # 最新の rewardの 平均値を all_period_reward_hashに 保持しておく
                    mean_val_stored_key = str(memory_hash[keyval[0]][0]) + str(memory_hash[keyval[0]][1])
                    try:
                        past_all_itr_mean_reward = all_period_reward_hash[mean_val_stored_key]
                    except:
                        past_all_itr_mean_reward = 0
                    current_itr_num = cur_itr + 1
                    # 過去の結果は最適な行動を学習する過程で見ると古い学習状態での値であるため
                    # 時間割引の考え方を導入して平均をとる
                    update_val = ((past_all_itr_mean_reward * (current_itr_num - 1) * gamma_at_reward_mean) + keyval[1]) / current_itr_num
                    print("update_reward: cur_itr=" + str(cur_itr) + " episode=" + str(episode) + " action=" + str(action) + " update_val=" + str(update_val))

                    memory_hash[keyval[0]][2] = update_val
                    all_period_reward_hash[mean_val_stored_key] = update_val

                    # memoryオブジェクトにはall_period_reward_arrの参照が渡してあるため
                    # memoryオブジェクト内の値も更新される
                    if keyval[3] == LONG: #BUY
                        all_period_reward_arr[keyval[2]][0] = update_val
                    else: #NOT_HAVE (DONOT)
                        all_period_reward_arr[keyval[2]][2] = update_val
            # CLOSEのrewardは必ず-100.0が返るようにしているため平均値を求める必要はない

            state = next_state  # 状態更新

            # Qネットワークの重みを学習・更新する replay
            # # memory無いの1要素でfitが行われるため、cur_idx=0から行ってしまって問題ない <- バッチ1はなんかアレなので今は変えている
            # batch_size分新たにmemoryにエピソードがたまったら batch_size のバッチとして replayする
            if episode + 1 >= batch_size and (episode + 1) % batch_size == 0:
                mainQN.replay(memory, time_series, cur_episode_idx=episode)

        # 一周回したら、次の周で利用されることはないのでクリア
        memory_hash = {}
        # 次周では過去のmemoryは参照しない（rewardは別途保持されている）ので、memoryはクリアしてしまう
        memory.clear()

def run_backtest(backtest_type):
    env_master = FXEnvironment(time_series=time_series, holdable_positions=holdable_positions)
    env = env_master.get_env(backtest_type)
    num_episodes = 1500000  # 10年. envがdoneを返すはずなので適当にでかい数字を設定しておく

    # [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    mainQN = QNetwork(learning_rate=learning_rate, time_series=time_series)     # メインのQネットワーク
    actor = Actor()

    mainQN.load_model("mainQN")

    # DONOT でスタート
    state, reward, done, info, needclose = env.step(0)
    #state = state.T
    state = np.reshape(state, [time_series, feature_num])
    for episode in range(num_episodes):   # 試行数分繰り返す
        if needclose:
            action = 1
        else:
            action = actor.get_action(state, episode, mainQN, 0, isBacktest = True)   # 時刻tでの行動を決定する

        state, reward, done, info, needclose  = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
        # 環境が提供する期間が最後までいった場合
        if done:
            print('all training period learned.')
            break
        #state = state.T
        state = np.reshape(state, [time_series, feature_num])

if __name__ == '__main__':
    np.random.seed(1337)  # for reproducibility
    if sys.argv[1] == "train":
        tarin_agent()
    elif sys.argv[1] == "backtest":
        run_backtest("backtest")
    elif sys.argv[1] == "backtest_test":
        run_backtest("backtest_test")
    else:
        print("please pass argument 'train' or 'backtest'")

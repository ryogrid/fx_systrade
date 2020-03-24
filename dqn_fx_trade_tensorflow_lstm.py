# coding:utf-8
# [0]必要なライブラリのインポート

# this code based on code on https://qiita.com/sugulu/items/bc7c70e6658f204f85f9
# I am very grateful to work of Mr. Yutaro Ogawa (id: sugulu)

IS_TF_STYLE = True #True #False
USE_TENSOR_BOARD = False
ENABLE_PRE_EXCUTION_OF_PREDICT = False
ENABLE_L2_LEGURALIZER = False
IS_PREDICT_BUY_DONOT_ONLY_MODE = True

import numpy as np
import tensorflow as tf
if IS_TF_STYLE:
    from tensorflow.keras.models import Sequential, model_from_json, Model, load_model, save_model
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LSTM, RepeatVector, TimeDistributed, Reshape, LeakyReLU
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras.regularizers import l2
    #from tensorflow.keras.regularizers import l2
    #from keras import backend as K
else:
    from keras.models import Sequential, model_from_json, Model, load_model
    from keras.layers import Dense, BatchNormalization, Dropout, LSTM, RepeatVector, TimeDistributed, Reshape, LeakyReLU
    from keras.optimizers import Adam, SGD
    from keras.regularizers import l2

from collections import deque
import pickle
from agent_fx_environment_lstm import FXEnvironment
import os
import sys
import random
import itertools
import math

# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.001, state_size=15, action_size=3, time_series=32):
        global all_period_reward_arr

        self.optimizer = Adam(lr=learning_rate, clipvalue=5.0)
        # self.optimizer = SGD(lr=learning_rate, momentum=0.9, clipvalue=5.0)
        self.loss_func = tf.keras.losses.Huber(delta=1.0)

        if IS_TF_STYLE:
            if ENABLE_L2_LEGURALIZER:
                self.model = tf.keras.Sequential([
                    LSTM(hidden_size, input_shape=(time_series, state_size), return_sequences=True, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01),
                                bias_regularizer=l2(0.01), activation=None), #, recurrent_dropout=0.5),
                    BatchNormalization(),
                    Dropout(0.5),
                    LeakyReLU(0.2),
                    LSTM(hidden_size, return_sequences=False, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01),
                                bias_regularizer=l2(0.01), activation=None), # , recurrent_dropout=0.5),
                    LeakyReLU(0.2),
                    Dense(action_size, activation='linear', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))
                ])
            else:
                self.model = tf.keras.Sequential([
                    LSTM(hidden_size, input_shape=(time_series, state_size), return_sequences=True), #activation=None), #recurrent_dropout=0.5),
                    BatchNormalization(),
                    LeakyReLU(0.2),
                    Dropout(0.5),
                    LSTM(hidden_size, return_sequences=False), # activation=None), #recurrent_dropout=0.5),
                    LeakyReLU(0.2),
                    Dense(action_size, activation='linear')
                ])
            self.model.compile(optimizer=self.optimizer, loss=self.loss_func)
        else:
            self.model = Sequential()

            if ENABLE_L2_LEGURALIZER:
                self.model.add(
                    LSTM(hidden_size, input_shape=(time_series, state_size), return_sequences=True, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01),
                                bias_regularizer=l2(0.01), activation=None)) #, recurrent_dropout=0.5))
                self.model.add(BatchNormalization())
                self.model.add(Dropout(0.5))
                self.model.add(LeakyReLU(0.2))
                self.model.add(LSTM(hidden_size, return_sequences=False, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation=None)) #,  recurrent_dropout=0.5))
                self.model.add(LeakyReLU(0.2))
                self.model.add(Dense(action_size, activation='linear', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            else:
                self.model.add(
                    LSTM(hidden_size, input_shape=(time_series, state_size), return_sequences=True, activation=None)) #, recurrent_dropout=0.5))
                self.model.add(BatchNormalization())
                self.model.add(Dropout(0.5))
                self.model.add(LeakyReLU(0.2))
                self.model.add(LSTM(hidden_size, return_sequences=False, activation=None)) #, recurrent_dropout=0.5))
                self.model.add(LeakyReLU(0.2))
                self.model.add(
                    Dense(action_size, activation='linear'))

            self.model.compile(optimizer=self.optimizer, loss=self.loss_func)

        self.model.summary()

        # # predictしたBUYとDONOTの報酬の絶対値の差分を保持する
        # self.buy_donot_diff_memory_predicted = Memory([], all_period_reward_arr = all_period_reward_arr, max_size=10000)

        # 配列で保持しているBUYとDONOTの報酬の平均値の絶対値の差分を保持する
        self.buy_donot_diff_memory_collect = Memory([], all_period_reward_arr = all_period_reward_arr, max_size=TRAIN_DATA_NUM)

        # self.batch_datas_for_generator = []

    # 重みの学習
    def replay(self, memory, time_series, cur_episode_idx = 0, batch_num = 1):
        inputs = np.zeros((batch_size * batch_num, time_series, feature_num))
        targets = np.zeros((batch_size * batch_num, 1, nn_output_size))

        all_sample_cnt = 0
        episode_idx = batch_size - 1 # 1引いているのは後続のコードがゼロオリジンを想定しているため
        if batch_num == 1:
            episode_idx = cur_episode_idx

        for ii in range(batch_num):
            mini_batch = memory.get_sequencial_samples(batch_size, (episode_idx + 1) - batch_size)
            # rewardだけ別管理の平均値のリストに置き換える
            mini_batch = memory.get_sequencial_converted_samples(mini_batch, (episode_idx + 1) - batch_size)

            for idx, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
                reshaped_state = np.reshape(state_b, [1, time_series, feature_num])
                inputs[all_sample_cnt] = reshaped_state

                # 学習の進行度の目安としてBUYとDONOTの predict した報酬の絶対値の差の平均値を求める
                # （理想的に学習していれば、両者は絶対値が同じ符号が逆の値になるはずであるので、0に近づくほど学習が進んでいると見なせる、はず）
                # residual は 残差 の意

                # targets[all_sample_cnt] = np.reshape(self.model.predict(reshaped_state)[0], [1, nn_output_size])
                # self.buy_donot_diff_memory_predicted.add(abs(abs(targets[all_sample_cnt][0][0]) - abs(targets[all_sample_cnt][0][2])))
                # agent_learn_residual = self.buy_donot_diff_memory_predicted.get_mean_value()

                # 正解の方を求めて出力
                self.buy_donot_diff_memory_collect.add_buy_donot_abs_diff(((episode_idx + 1) - batch_size) + idx)
                base_data_residual = self.buy_donot_diff_memory_collect.get_mean_value()

                print("reward_b: collect residual -> " + str(base_data_residual))

                # イテレーションをまたいで平均rewardを計算しているlistから3つ全てのアクションのrewardを得てあるので
                # 全て設定する
                # BUYとDONOTの教師信号は符号で-1, 1 にクリッピングする
                if IS_PREDICT_BUY_DONOT_ONLY_MODE:
                    targets[all_sample_cnt][0][0] = 1.0 if reward_b[0] > 0 else -1.0 # 教師信号
                    targets[all_sample_cnt][0][1] = 1.0 if reward_b[2] > 0 else -1.0  # 教師信号
                else:
                    targets[all_sample_cnt][0][0] = 1.0 if reward_b[0] > 0 else -1.0 # 教師信号
                    targets[all_sample_cnt][0][1] = -100.0  # CLOSEのrewardは必ず-100.0
                    targets[all_sample_cnt][0][2] = 1.0 if reward_b[2] > 0 else -1.0  # 教師信号

                all_sample_cnt += 1

            episode_idx += batch_size

        targets = np.array(targets)
        inputs = np.array(inputs)

        inputs = inputs.reshape((batch_size * batch_num, time_series, feature_num))
        targets = targets.reshape((batch_size * batch_num, nn_output_size))

        if IS_TF_STYLE:
            #self.fit(inputs, targets, epochs=1, verbose=1, batch_size=batch_size)
            cbks = []
            if USE_TENSOR_BOARD:
                # tensorboardのためのデータを記録するコールバック
                callbacks = tf.keras.callbacks.TensorBoard(log_dir='logdir', histogram_freq=1,
                                                        write_graph=True, write_grads=True, profile_batch=True)
                cbks = [callbacks]
            self.model.fit(inputs, targets, epochs=1, verbose=1, batch_size=batch_size, callbacks=cbks)
        else:
            self.model.fit(inputs, targets, epochs=1, verbose=1, batch_size=batch_size)

        #     # 複数コアで並列に処理するため、複数バッチが貯まったら git_generatorで fit を行う
        #     # Windows環境では動作しない
        #     def batch_generator():
        #         nonlocal self
        #         for idx, (x, y) in enumerate(self.batch_datas_for_generator):
        #             yield (x, y)
        #
        #     self.batch_datas_for_generator.append([inputs, targets])
        #     if len(self.batch_datas_for_generator) >= self.batch_num_to_call_fit:
        #         self.model.fit_generator(batch_generator(), epochs=1, shuffle=False, workers=8, use_multiprocessing=True, verbose=1, steps_per_epoch=self.batch_num_to_call_fit)
        #         self.batch_datas_for_generator = []

    def fit(self, inputs, targets, epochs=1, verbose=0, batch_size=32):
        # config = tf.ConfigProto(
        #     gpu_options=tf.GPUOptions(
        #         visible_device_list="0",  # specify GPU number
        #         allow_growth=True
        #     )
        # )
        # sess = tf.Session(config=config)
        # sess.run(tf.global_variables_initializer())

        #with sess:
        bat_per_epoch = math.floor(len(inputs) / batch_size)
        for epoch in range(epochs):
            for i in range(bat_per_epoch):
                n = i * batch_size
                self.fit_step(inputs[n:n + batch_size], targets[n:n + batch_size])

    # 入力データはバッチ単位で与えられる
    def fit_step(self, input, target, epochs=1, verbose=0, batch_size=32):
        with tf.GradientTape() as tape:
            predicted = self.model(input, training=True)
            # # assertを入れて出力の型をチェックする。
            # tf.debugging.assert_equal(logits.shape, (32, 10))
            loss_value = self.loss_func(target, predicted)
            print("loss: " + str(loss_value))

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def save_model(self, file_path_prefix_str):
        if IS_TF_STYLE:
            save_model(self.model, "./" + file_path_prefix_str + ".hd5", save_format="h5")
        else:
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
    def __init__(self):
        self.init_selected_action_q()

    def init_selected_action_q(self):
        self.pre_selected_action_q = deque([], maxlen=(HODABLE_POSITIONS * 2 + 10))

    def generate_pre_selected_actions(self, q_network, generate_num, experienced_episodes, episode_idx, state_arr):
        generated_action_arr = [-1] * generate_num
        states_to_predict = []
        # -1 しているのは イテレーションループが始まる前に一度アクションをとっていることとのつじつま合わせ
        episode_idx_conved_to_sate_idx = episode_idx + time_series - 1
        epsilon = 0.001 + 0.9 / (1.0 + (300.0 * (experienced_episodes / TOTAL_ACTION_NUM)))
        #epsilon = 0.001
        for idx in range(generate_num):
            if epsilon <= np.random.uniform(0, 1):
                # +1 しているのは 現時点でのインデックスの state を含めるため（rangeは第二引数の値は返さないので）
                states_to_predict.append(state_arr[episode_idx_conved_to_sate_idx - time_series + 1:episode_idx_conved_to_sate_idx + 1])
                generated_action_arr[idx] = None
            else:
                if IS_PREDICT_BUY_DONOT_ONLY_MODE:
                    # ランダムに行動する
                    # 現在の実装ではagentが自発的にCLOSEを選択することはないので、BUYかDONOTの2つからランダム選択する
                    # (PREDICT_BY_DNOT_ONLY_MODEでは、 DONOTがpredictの結果としては1として返ることを想定している）
                    action = np.random.choice([0, 1])
                    generated_action_arr[idx] = action
                else:
                    # ランダムに行動する
                    # 現在の実装ではagentが自発的にCLOSEを選択することはないので、BUYかDONOTの2つからランダム選択する
                    action = np.random.choice([0, 2])
                    generated_action_arr[idx] = action

        # generated_action_arrのNoneになっている要素をpredictして値を埋める

        print("elem num of states_to_predict: " + str(len(states_to_predict)))
        if len(states_to_predict) > 0:
            reshaped_state = np.reshape(states_to_predict, [len(states_to_predict), time_series, feature_num])
            predicted_values = q_network.model.predict(reshaped_state)
            #print(predicted_values)
            predicted_values_q = deque([], maxlen=generate_num)
            # 配列だと扱いずらいのでキューに移し替える
            for idx in range(len(predicted_values)):
                #print(predicted_values[idx])
                predicted_values_q.append(predicted_values[idx])
            for idx in range(len(generated_action_arr)):
                if generated_action_arr[idx] == None:
                    predicted_val = predicted_values_q.pop()
                    #print(predicted_val)
                    #print("NN all output at get_action: " + str(list(itertools.chain.from_iterable(predicted_val))))
                    print("NN all output at get_action: " + str(predicted_val))
                    generated_action_arr[idx] = np.argmax(predicted_val)

        # 生成したリストをキューの末尾にまとめて追加する
        self.pre_selected_action_q.extend(generated_action_arr)

    def get_action(self, state, experienced_episodes, mainQN, episode_idx, isBacktest = False, buy_num = None, donot_num = None, state_arr = None):   # [C]ｔ＋１での行動を返す
        # 可能な限りpredictをまとめて発行しておくモード
        # (CLOSEが発生するまでは、同一イテレーションの他の選択とは独立にBUYかDONOTを選択できる点を活用する)
        if ENABLE_PRE_EXCUTION_OF_PREDICT and isBacktest == False:
            #print("episode_idx:" + str(episode_idx))
            #print("buy_num:" + str(buy_num))
            #print("donot_num:" + str(donot_num))
            action = -1
            try:
                action = self.pre_selected_action_q.pop()
            except:
                #生成しておいたアクションが無くなったので再生成する
                generate_action_num = min([HODABLE_POSITIONS - buy_num, HODABLE_POSITIONS - donot_num])
                print("generate_action_num:" + str(generate_action_num))
                # アクションを生成しておく
                self.generate_pre_selected_actions(mainQN, generate_action_num, experienced_episodes, episode_idx, state_arr)

                action = self.pre_selected_action_q.pop()
                if IS_PREDICT_BUY_DONOT_ONLY_MODE:
                    # このモードではCLOSEをpredictさせず、BUYとDONOTの報酬のみ扱う
                    # 従って、0, 1 が返ってくるが、 0, 2 の値を返す必要がある
                    # 以下はそのためのつじつま合わせ
                    action = 2* action

            return action

        else: #通常モード
            # 徐々に最適行動のみをとる、ε-greedy法
            epsilon = 0.001 + 0.9 / (1.0 + (300.0 * (experienced_episodes / TOTAL_ACTION_NUM)))

            # epsilonが小さい値の場合の方が最大報酬の行動が起こる
            # イテレーション数が5の倍数の時か、バックテストの場合は常に最大報酬の行動を選ぶ
            if epsilon <= np.random.uniform(0, 1) or isBacktest == True:
                reshaped_state = np.reshape(state, [1, time_series, feature_num])
                retTargetQs = mainQN.model.predict(reshaped_state)
                print("NN all output at get_action: " + str(list(itertools.chain.from_iterable(retTargetQs))))
                action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
                if IS_PREDICT_BUY_DONOT_ONLY_MODE:
                    # このモードではCLOSEをpredictさせず、BUYとDONOTの報酬のみ扱う
                    # 従って、0, 1 が返ってくるが、 0, 2 の値を返す必要がある
                    # 以下はそのためのつじつま合わせ
                    action = 2* action
            else:
                # ランダムに行動する
                # 現在の実装ではagentが自発的にCLOSEを選択することはないので、BUYかDONOTの2つからランダム選択する
                action = np.random.choice([0, 1, 2])
                #action = np.random.choice([0, 2])

            return action

# [5] メイン関数開始----------------------------------------------------
# [5.1] 初期設定--------------------------------------------------------

# ---
#gamma = 0.95 # <- 今の実装では利用されていない #0.99 #0.3 # #0.99 #0.3 #0.99  # 割引係数
hidden_size = 64 #32 #24 #50 #28 #80 #28 #50 # <- 50層だとバッチサイズ=32のepoch=1で1エピソード約3時間かかっていた # Q-networkの隠れ層のニューロンの数
learning_rate = 0.0004 #0.0016 #0.0001 #0.01 #0.001 #0.01 #0.0005 # 0.0005 #0.0001 #0.005 #0.01 # 0.05 #0.001 #0.0001 # 0.00001         # Q-networkの学習係数
time_series = 32 #64 #32 #64 #32
batch_size = 256 #1024 #64 #8 #64 #8 #1 #64 #16 #32 #16 #32 #64 # 32  # Q-networkを更新するバッチの大きさ
TRAIN_DATA_NUM = 72000 - time_series # <- 5分足で1年 #36000 - time_series # <- 10分足で1年  #12000 - time_series # <- 30分足で1年 #72000 - time_series #36000 - time_series # 1000 - time_series #テストデータでうまくいくまで半年に減らす  #74651 # <- 検証中は期間を1年程度に減らす　223954 # 3years (test is 5 years)
num_episodes = TRAIN_DATA_NUM + 10  # envがdoneを返すはずなので念のため多めに設定 #1000  # 総試行回数
iteration_num = 5000 #720 # <- 劇的に減らす(1足あたり 16 * 1 * 50 で800回のfitが行われる計算) #720 #20
#memory_size = TRAIN_DATA_NUM * iteration_num + 10 #TRAIN_DATA_NUM * int(iteration_num * 0.2) # 全体の20%は収まるサイズ. つまり終盤は最新の当該割合に対応するエピソードのみreplayする #10000
memory_size = TRAIN_DATA_NUM * 2 + 10 #TRAIN_DATA_NUM * int(iteration_num * 0.2) # 全体の20%は収まるサイズ. つまり終盤は最新の当該割合に対応するエピソードのみreplayする #10000
feature_num = 9 #10 + 1 #10 + 9*3 #10 #11 #10 #11 #10 #11
if IS_PREDICT_BUY_DONOT_ONLY_MODE:
    nn_output_size = 2
else:
    nn_output_size = 3
TOTAL_ACTION_NUM = TRAIN_DATA_NUM * iteration_num
HODABLE_POSITIONS = 100 #30 # 100
BACKTEST_ITR_PERIOD = 30

# イテレーションを跨いで、ある足での action に対する reward の平均値を求める際に持ちいる時間割引率
# 昔に得られた結果だからといって割引してはCLOSEのタイミングごとに平等に反映されないことになるので
# 現在の実装では 1.0 とする
gamma_at_reward_mean = 1.0 #0.9

LONG = 0 #BUY
SHORT = 1 #CLOSE
NOT_HAVE = 2 #DONOT

all_period_reward_arr = [[0.0, -100.0, 0.0] for i in range(TRAIN_DATA_NUM)]

def tarin_agent():
    global all_period_reward_arr

    env_master = FXEnvironment(time_series=time_series, holdable_positions=HODABLE_POSITIONS)
    # [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    mainQN = QNetwork(time_series=time_series, learning_rate=learning_rate, state_size=feature_num, action_size=nn_output_size)     # メインのQネットワーク
    # mainQN_GPU = QNetwork(time_series=time_series, learning_rate=learning_rate, state_size=feature_num,
    #                   action_size=nn_output_size)  # メインのQネットワーク

    memory = Memory([], max_size=memory_size, all_period_reward_arr=all_period_reward_arr)
    memory_hash = {}
    actor = Actor()

    total_get_acton_cnt = 1

    #アクション選択の事前生成のための情報を保持する配列
    buy_num = 0
    donot_num = 0

    # if os.path.exists("./mainQN_nw.json"):
    if os.path.exists("./mainQN.hd5"):
        mainQN.load_model("mainQN")
        # memory.load_memory("memory")
        # with open("./total_get_action_count.pickle", 'rb') as f:
        #     total_get_acton_cnt = pickle.load(f)
        # with open("./all_period_reward_arr.pickle", 'rb') as f:
        #     all_period_reward_arr = pickle.load(f)

    def store_episode_log_to_memory(state, action, reward, next_state, info):
        nonlocal memory
        nonlocal memory_hash
        # nonlocal state_x_action_hash
        a_log = [state, action, reward, next_state]
        memory.add(a_log)  # メモリを更新する
        # 後からrewardを更新するためにエピソード識別子をキーにエピソードを取得可能としておく
        memory_hash[info[0]] = a_log

    #######################################################

    for cur_itr in range(iteration_num):
        # 定期的にバックテストを行い評価できるようにしておく（CSVを吐く）
        if cur_itr % BACKTEST_ITR_PERIOD == 0 and cur_itr != 0:
            run_backtest("auto_backtest", learingQN=mainQN)
            run_backtest("auto_backtest_test", learingQN=mainQN)
            continue

        env = env_master.get_env('train')
        #action = np.random.choice([0, 1, 2])
        action = np.random.choice([0, 2])
        if action == 0:
            buy_num += 1
        elif action == 2:
            donot_num += 1
        state, reward, done, info, needclose = env.step(action)  # 1step目は適当なBUYかDONOTのうちランダムに行動をとる
        total_get_acton_cnt += 1
        state = np.reshape(state, [time_series, feature_num])  # list型のstateを、1行15列の行列に変換
        # ここだけ 同じstateから同じstateに遷移したことにする
        store_episode_log_to_memory(state, action, reward, state, info)

        # スナップショットをとっておく
        if cur_itr % 20 == 0 and cur_itr != 0:
            mainQN.save_model("mainQN")
            # memory.save_memory("memory")
            # with open("./total_get_action_count.pickle", 'wb') as f:
            #     pickle.dump(total_get_acton_cnt, f)
            # with open("./all_period_reward_arr.pickle", 'wb') as f:
            #     pickle.dump(all_period_reward_arr, f)

        # replay呼び出しに用いる（上ですでに一回行っているので1からスタート）
        total_episode_on_last_itr = 1

        for episode in range(num_episodes):  # 試行数分繰り返す
            #with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
            # フィードするデータを用意している間はGPUは利用せず、CPU（コア）も一つのみとして動作させる
            total_get_acton_cnt += 1
            total_episode_on_last_itr += 1

            if needclose:
                action = 1
                # CLOSEが発生したので、事前生成しておいたものはクリアする
                actor.init_selected_action_q()
                buy_num = donot_num = 0
            else:
                # 時刻tでの行動を決定する
                action = actor.get_action(state, total_get_acton_cnt, mainQN, episode, buy_num=buy_num, donot_num=donot_num, state_arr=env.input_arr[env.time_series - 1:])
                if action == 0:
                    buy_num += 1
                elif action == 2:
                    donot_num += 1
                else: # 1 (CLOSE)
                    #基本的には選択されないはずだが、NNの学習中に選択されてしまう可能性はあるのでその場合の対処
                    actor.init_selected_action_q()
                    buy_num = donot_num = 0

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
                    past_all_itr_mean_reward = all_period_reward_arr[keyval[2]][keyval[3]]
                    current_itr_num = cur_itr + 1
                    # 過去の結果は最適な行動を学習する過程で見ると古い学習状態での値であるため
                    # 時間割引の考え方を導入して平均をとる
                    update_val = ((past_all_itr_mean_reward * (current_itr_num - 1) * gamma_at_reward_mean) + keyval[1]) / current_itr_num
                    print("update_reward: cur_itr=" + str(cur_itr) + " episode=" + str(episode) + " update_idx=" + str(keyval[2]) +  " action=" + str(keyval[3]) + " update_val=" + str(update_val))

                    memory_hash[keyval[0]][2] = update_val

                    # memoryオブジェクトにはall_period_reward_arrの参照が渡してあるため
                    # memoryオブジェクト内の値も更新される
                    if keyval[3] == LONG: #BUY
                        all_period_reward_arr[keyval[2]][0] = update_val
                    else: #NOT_HAVE (DONOT)
                        all_period_reward_arr[keyval[2]][2] = update_val
            # CLOSEのrewardは必ず-100.0が返るようにしているため平均値を求める必要はない

            state = next_state  # 状態更新

            # # Qネットワークの重みを学習・更新する replay
            # # # memory無いの1要素でfitが行われるため、cur_idx=0から行ってしまって問題ない <- バッチ1はなんかアレなので今は変えている
            # # batch_size分新たにmemoryにエピソードがたまったら batch_size のバッチとして replayする
            # if episode + 1 >= batch_size and (episode + 1) % batch_size == 0:
            #     mainQN.replay(memory, time_series, cur_episode_idx=episode)

        # イテレーションの最後にまとめて複数ミニバッチでfitする
        # これにより、fitがコア並列やGPUで動作していた場合のオーバヘッド削減を狙う
        #with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        mainQN.replay(memory, time_series, cur_episode_idx=0, batch_num=(total_episode_on_last_itr // batch_size))

        # 一周回したら、次の周で利用されることはないのでクリア
        memory_hash = {}
        # 次周では過去のmemoryは参照しない（rewardは別途保持されている）ので、memoryはクリアしてしまう
        memory.clear()
        # 次週では空の状態でスタート
        actor.init_selected_action_q()
        # 次週ではリセット
        buy_num = donot_num = 0

def run_backtest(backtest_type, learingQN=None):
    env_master = FXEnvironment(time_series=time_series, holdable_positions=HODABLE_POSITIONS)
    env = env_master.get_env(backtest_type)
    num_episodes = 1500000  # 10年. envがdoneを返すはずなので適当にでかい数字を設定しておく

    # [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    mainQN = QNetwork(learning_rate=learning_rate, time_series=time_series)     # メインのQネットワーク
    actor = Actor()

    if backtest_type == "auto_backtest" or backtest_type == "auto_backtest_test":
        mainQN = learingQN
    else:
        mainQN.load_model("mainQN")

    # DONOT でスタート
    state, reward, done, info, needclose = env.step(0)
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

def disable_gpu():
    tf.config.set_visible_devices([], 'GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
    print(logical_devices)

def disable_multicore():
    physical_devices = tf.config.list_physical_devices('CPU')
    try:
        # Disable first GPU
        tf.config.set_visible_devices(physical_devices[0], 'CPU')
        logical_devices = tf.config.list_logical_devices('CPU')
        print(logical_devices)
        # Logical device was not created for first GPU
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

def limit_gpu_memory_usage():
    # GPUのGPUメモリ使用量にリミットをかける
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

if __name__ == '__main__':
    np.random.seed(1337)  # for reproducibility

    # TensorFlowの低レイヤ寄りの機能群を利用した実装だと、現状GPUがあるとエラーになるため
    # GPU自体が存在しないように見えるようにしておく
    # また、バックテストだけ行う際もGPUで predictすると遅いので搭載されてないものとして動作させる
    if IS_TF_STYLE:
        #disable_multicore()
        disable_gpu()
        #limit_gpu_memory_usage()
    elif sys.argv[1] == "train" or sys.argv[1] == "backtest" or sys.argv[1] == "backtest_test":
        #disable_multicore()
        disable_gpu()
        #limit_gpu_memory_usage()

    if sys.argv[1] == "train":
        tarin_agent()
    elif sys.argv[1] == "backtest":
        run_backtest("backtest")
    elif sys.argv[1] == "backtest_test":
        run_backtest("backtest_test")
    else:
        print("please pass argument 'train' or 'backtest'")

# coding:utf-8
# [0]必要なライブラリのインポート

# this code based on code on https://qiita.com/sugulu/items/bc7c70e6658f204f85f9
# I am very grateful to work of Mr. Yutaro Ogawa (id: sugulu)

#IS_TF_STYLE = True #True #False
USE_TENSOR_BOARD = False
#ENABLE_PRE_EXCUTION_OF_PREDICT = False
#ENABLE_L2_LEGURALIZER = False
#IS_PREDICT_BUY_DONOT_ONLY_MODE = True

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, model_from_json, Model, load_model, save_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LSTM, RepeatVector, TimeDistributed, Reshape, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1, l2
#from tensorflow.keras.regularizers import l2

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

        self.optimizer = Adam(lr=learning_rate, clipvalue=0.5)
        #self.optimizer = RMSprop(lr=learning_rate, momentum=0.9, clipvalue=0.1)
        #self.optimizer = SGD(lr=learning_rate, momentum=0.9, clipvalue=0.5)
        #self.loss_func = tf.keras.losses.Huber(delta=1.0)
        self.loss_func = "categorical_crossentropy"

        self.model = tf.keras.Sequential([
            LSTM(hidden_size, input_shape=(time_series, state_size), return_sequences=True, activation=None), #kernel_regularizer=l1(0.1)), #recurrent_dropout=0.5),
            LeakyReLU(0.2),
            #PReLU(),
            #BatchNormalization(),
            #Dropout(0.5),
            LSTM(hidden_size, return_sequences=False, activation=None), #kernel_regularizer=l1(0.1)), #recurrent_dropout=0.5),
            LeakyReLU(0.2),
            #PReLU(),
            BatchNormalization(),
            #Dropout(0.5),
            Dense(action_size, activation='softmax')
            #Dense(action_size, activation='linear')
        ])
        self.model.compile(optimizer=self.optimizer, loss=self.loss_func, metrics=['accuracy'])
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


                # # イテレーションをまたいで平均rewardを計算しているlistから3つ全てのアクションのrewardを得てあるので
                # # 全て設定する
                # # BUYとDONOTの教師信号は符号で-1, 1 にクリッピングする
                # targets[all_sample_cnt][0][0] = 1.0 if reward_b[0] > 0 else -1.0 # 教師信号
                # targets[all_sample_cnt][0][1] = 1.0 if reward_b[2] > 0 else -1.0  # 教師信号

                bigger_pips_action = np.argmax(reward_b)
                if bigger_pips_action == 0:
                    targets[all_sample_cnt][0][0] = 1 # 教師信号
                    targets[all_sample_cnt][0][1] = 0 # 教師信号
                else: # 2 => DONOT
                    targets[all_sample_cnt][0][0] = 0 # 教師信号
                    targets[all_sample_cnt][0][1] = 1 # 教師信号

                all_sample_cnt += 1

            episode_idx += batch_size

        targets = np.array(targets)
        inputs = np.array(inputs)

        inputs = inputs.reshape((batch_size * batch_num, time_series, feature_num))
        targets = targets.reshape((batch_size * batch_num, nn_output_size))

        cbks = []
        if USE_TENSOR_BOARD:
            # tensorboardのためのデータを記録するコールバック
            callbacks = tf.keras.callbacks.TensorBoard(log_dir='logdir', histogram_freq=1,
                                                    write_graph=True, write_grads=True, profile_batch=True)
            cbks = [callbacks]
        self.model.fit(inputs, targets, epochs=1, verbose=1, batch_size=batch_size, callbacks=cbks)
        # self.fit(inputs, targets, epochs=1, verbose=1, batch_size=batch_size)

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
        save_model(self.model, "./" + file_path_prefix_str + ".hd5", save_format="h5")
        # with open("./" + file_path_prefix_str + "_nw.json", "w") as f:
        #     f.write(self.model.to_json())
        # self.model.save_weights("./" + file_path_prefix_str + "_weights.hd5")

    def load_model(self, file_path_prefix_str):
        self.model = load_model("./" + file_path_prefix_str + ".hd5", compile=False)
        # with open("./" + file_path_prefix_str + "_nw.json", "r") as f:
        #     self.model = model_from_json(f.read())
        # self.model.compile(loss=huberloss, optimizer=self.optimizer)
        # self.model.load_weights("./" + file_path_prefix_str + "_weights.hd5")

# replay に 利用するための キュー的なもの
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
        pass

    def get_action(self, state, experienced_episodes, mainQN, episode_idx, isBacktest = False):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0 + (300.0 * (experienced_episodes / TOTAL_ACTION_NUM)))

        # epsilonが小さい値の場合の方が最大報酬の行動が起こる
        # イテレーション数が5の倍数の時か、バックテストの場合は常に最大報酬の行動を選ぶ
        if epsilon <= np.random.uniform(0, 1) or isBacktest == True:
            reshaped_state = np.reshape(state, [1, time_series, feature_num])
            retTargetQs = mainQN.model.predict(reshaped_state)
            print("NN all output at get_action: " + str(list(itertools.chain.from_iterable(retTargetQs))))
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
            # CLOSEをpredictさせず、BUYとDONOTの報酬のみ扱っている
            # 従って、0, 1 が返ってくるが、 0, 2 の値を返す必要がある
            # 以下はそのためのつじつま合わせ
            action = 2* action
        else:
            # ランダムに行動する
            action = np.random.choice([0, 1, 2])

        return action

# ---
hidden_size = 64 #32
learning_rate = 0.0001 #0.0016
time_series = 64 #32
batch_size = 64 #256 #1024
TRAIN_DATA_NUM = 72000 # <- 5分足で1年
num_episodes = TRAIN_DATA_NUM + 10  # envがdoneを返すはずなので念のため多めに設定
iteration_num = 5000 #720
memory_size = TRAIN_DATA_NUM * 2 + 10
feature_num = 10
nn_output_size = 2
TOTAL_ACTION_NUM = TRAIN_DATA_NUM * iteration_num
HODABLE_POSITIONS = 100 #30
BACKTEST_ITR_PERIOD = 30
half_spread = 0.0015

# イテレーションを跨いで、ある足での action に対する reward の平均値を求める際に持ちいる時間割引率
# 昔に得られた結果だからといって割引してはCLOSEのタイミングごとに平等に反映されないことになるので
# 現在の実装では 1.0 とする
gamma_at_reward_mean = 1.0
# mean_pips_update_stop_itr = 90

LONG = 0 #BUY
SHORT = 1 #CLOSE
NOT_HAVE = 2 #DONOT


all_period_reward_arr = [[0.0, -100.0, 0.0] for i in range(TRAIN_DATA_NUM)]

def tarin_agent():
    global all_period_reward_arr

    env_master = FXEnvironment(TRAIN_DATA_NUM, time_series=time_series, holdable_positions=HODABLE_POSITIONS, half_spread = half_spread)
    mainQN = QNetwork(time_series=time_series, learning_rate=learning_rate, state_size=feature_num, action_size=nn_output_size)     # メインのQネットワーク
    # mainQN_GPU = QNetwork(time_series=time_series, learning_rate=learning_rate, state_size=feature_num,
    #                   action_size=nn_output_size)  # メインのQネットワーク

    memory = Memory([], max_size=memory_size, all_period_reward_arr=all_period_reward_arr)
    memory_hash = {}
    actor = Actor()

    total_get_acton_cnt = 1

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
            mainQN.save_model("mainQN")
            run_backtest("auto_backtest", learingQN=mainQN, env_master=env_master)
            run_backtest("auto_backtest_test", learingQN=mainQN, env_master=env_master)
            continue

        env = env_master.get_env('train')
        #action = np.random.choice([0, 1, 2])
        action = np.random.choice([0, 2])
        state, reward, done, info, needclose = env.step(action)  # 1step目は適当なBUYかDONOTのうちランダムに行動をとる
        total_get_acton_cnt += 1
        state = np.reshape(state, [time_series, feature_num])  # list型のstateを、1行15列の行列に変換
        # ここだけ 同じstateから同じstateに遷移したことにする
        store_episode_log_to_memory(state, action, reward, state, info)

        # # スナップショットをとっておく
        # if cur_itr % 20 == 0 and cur_itr != 0:
        #     mainQN.save_model("mainQN")
        #     # memory.save_memory("memory")
        #     # with open("./total_get_action_count.pickle", 'wb') as f:
        #     #     pickle.dump(total_get_acton_cnt, f)
        #     # with open("./all_period_reward_arr.pickle", 'wb') as f:
        #     #     pickle.dump(all_period_reward_arr, f)

        # replay呼び出しに用いる（上ですでに一回行っているので1からスタート）
        total_episode_on_last_itr = 1

        for episode in range(num_episodes):  # 試行数分繰り返す
            #with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
            # フィードするデータを用意している間はGPUは利用せず、CPU（コア）も一つのみとして動作させる
            total_get_acton_cnt += 1
            total_episode_on_last_itr += 1

            if needclose:
                action = 1
            # elif cur_itr > mean_pips_update_stop_itr:
            #     action = np.random.choice([0, 2])
            else:
                # 時刻tでの行動を決定する
                action = actor.get_action(state, total_get_acton_cnt, mainQN, episode)

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
            if len(info) > 1: #and cur_itr <= mean_pips_update_stop_itr:
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
        mainQN.replay(memory, time_series, cur_episode_idx=0, batch_num=(total_episode_on_last_itr // batch_size))

        # 一周回したら、次の周で利用されることはないのでクリア
        memory_hash = {}
        # 次周では過去のmemoryは参照しない（rewardは別途保持されている）ので、memoryはクリアしてしまう
        memory.clear()

def run_backtest(backtest_type, learingQN=None, env_master=None):
    if env_master:
        env_master_local = env_master
    else:
        env_master_local = FXEnvironment(TRAIN_DATA_NUM, time_series=time_series, holdable_positions=HODABLE_POSITIONS, half_spread=half_spread)

    env = env_master_local.get_env(backtest_type)
    num_episodes = 1500000  # 10年. envがdoneを返すはずなので適当にでかい数字を設定しておく

    mainQN = QNetwork(learning_rate=learning_rate, time_series=time_series)     # メインのQネットワーク
    actor = Actor()

    mainQN.load_model("mainQN")
    # if backtest_type == "auto_backtest" or backtest_type == "auto_backtest_test":
    #     mainQN = learingQN
    # else:
    #     mainQN.load_model("mainQN")

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

# def disable_multicore():
#     physical_devices = tf.config.list_physical_devices('CPU')
#     try:
#         # Disable first GPU
#         tf.config.set_visible_devices(physical_devices[0], 'CPU')
#         logical_devices = tf.config.list_logical_devices('CPU')
#         print(logical_devices)
#         # Logical device was not created for first GPU
#     except:
#         # Invalid device or cannot modify virtual devices once initialized.
#         pass

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
    # 再現性のため乱数シードを固定
    random.seed(1337)
    np.random.seed(1337)
    tf.random.set_seed(1337)

    # GPUを用いると学習結果の再現性が担保できないため利用しない
    # また、バックテストだけ行う際もGPUで predictすると遅いので搭載されてないものとして動作させる
    disable_gpu()
    # limit_gpu_memory_usage()

    if sys.argv[1] == "train":
        tarin_agent()
    elif sys.argv[1] == "backtest":
        run_backtest("backtest")
    elif sys.argv[1] == "backtest_test":
        run_backtest("backtest_test")
    else:
        print("please pass argument 'train' or 'backtest'")

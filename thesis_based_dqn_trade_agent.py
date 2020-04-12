# coding:utf-8
# [0]必要なライブラリのインポート

# this code based on code on https://qiita.com/sugulu/items/bc7c70e6658f204f85f9
# I am very grateful to work of Mr. Yutaro Ogawa (id: sugulu)

USE_TENSOR_BOARD = False


import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, model_from_json, Model, load_model, save_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LSTM, RepeatVector, TimeDistributed, Reshape, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1, l2
#from tensorflow.keras.regularizers import l2

from collections import deque
import pickle
from thesis_based_dqn_trade_environment import FXEnvironment
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
        self.loss_func = tf.keras.losses.Huber(delta=1.0)
        #self.loss_func = "categorical_crossentropy"

        # self.model = tf.keras.Sequential([
        #     LSTM(hidden_size_lstm1, input_shape=(time_series, state_size), return_sequences=True, activation=None), #kernel_regularizer=l1(0.1)), #recurrent_dropout=0.5),
        #     LeakyReLU(0.2),
        #     #BatchNormalization(),
        #     #Dropout(0.5),
        #     LSTM(hidden_size_lstm2, return_sequences=False, activation=None), #kernel_regularizer=l1(0.1)), #recurrent_dropout=0.5),
        #     LeakyReLU(0.2),
        #     #BatchNormalization(),
        #     #Dropout(0.5),
        #     #Dense(action_size, activation='softmax')
        #     Dense(action_size, activation='linear')
        # ])
        self.model = tf.keras.Sequential([
            LSTM(hidden_size_lstm1, input_shape=(time_series, state_size), return_sequences=False, activation=None),
            LeakyReLU(0.2),
            BatchNormalization(),
            Dropout(0.5),
            Dense(hidden_size_dense1, activation=None),
            LeakyReLU(0.2),
            BatchNormalization(),
            Dropout(0.5),
            Dense(hidden_size_dense2, activation=None),
            LeakyReLU(0.2),
            Dropout(0.5),
            Dense(action_size, activation='linear')
        ])

        #self.model.compile(optimizer=self.optimizer, loss=self.loss_func, metrics=['accuracy'])
        self.model.compile(optimizer=self.optimizer, loss=self.loss_func)
        self.model.summary()

    # 重みの学習
    def replay(self, memory, time_series, targetQN, cur_episode_idx = 0, batch_num = 1):
        inputs = np.zeros((batch_size * batch_num, time_series, feature_num))
        targets = np.zeros((batch_size * batch_num, 1, nn_output_size))

        all_sample_cnt = 0
        #batch_start_idx = (cur_episode_idx + 1) - (batch_size * batch_num)
        # ミニバッチは後方から取得して targets に詰めていく
        # ミニバッチの中は時系列になっているが、ミニバッチ単位では時系列が逆になる
        # これは、後方から前方にrewardが伝播していく更新式の性質を考慮したため（うまくいかなければやめる）
        batch_start_idx = (cur_episode_idx + 1) - batch_size

        for ii in range(batch_num):
            mini_batch = memory.get_sequencial_samples(batch_size, batch_start_idx)

            for idx, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
                reshaped_state = np.reshape(state_b, [1, time_series, feature_num])
                inputs[all_sample_cnt] = reshaped_state

                # Double DQN (mainQNとtargetQNを用いる。 Fixed Q-targetsもこれでおそらく実現できているのではないかと思われる)
                reshaped_next_state = np.reshape(next_state_b, [1, time_series, feature_num])
                retmainQs = self.model.predict(reshaped_next_state)[0]
                next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                predicted_targetQN = targetQN.model.predict(reshaped_next_state)
                target = reward_b + gamma * predicted_targetQN[0][next_action]

                targets[all_sample_cnt][0] = self.model.predict(reshaped_state)[0]
                targets[all_sample_cnt][0][action_b] = target  # 教師信号

                # bigger_pips_action = np.argmax(reward_b)
                # if bigger_pips_action == 0:
                #     targets[all_sample_cnt][0][0] = 1 # 教師信号
                #     targets[all_sample_cnt][0][1] = 0 # 教師信号
                # else: # 2 => DONOT
                #     targets[all_sample_cnt][0][0] = 0 # 教師信号
                #     targets[all_sample_cnt][0][1] = 1 # 教師信号

                print("reward_b," + str(reward_b) + ",target," + str(target) + ",action," + str(action_b) + ",next_action," + str(next_action))

                all_sample_cnt += 1

            #batch_start_idx += batch_size
            batch_start_idx -= batch_size

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

    def save_model(self, file_path_prefix_str):
        save_model(self.model, "./" + file_path_prefix_str + ".hd5", save_format="h5")

    def load_model(self, file_path_prefix_str):
        self.model = load_model("./" + file_path_prefix_str + ".hd5", compile=False)

# replay に 利用するための キュー的なもの
class Memory:
    def __init__(self, initial_elements, max_size=1000):
        self.max_size = max_size
        self.buffer = deque(initial_elements, maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def get_last(self, num):
        deque_length = len(self.buffer)
        start = deque_length - num
        end = deque_length
        return [self.buffer[ii] for ii in range(start, end)]

    # 呼び出し側がmemory内の適切なstart要素インデックスを計算して呼び出す
    def get_sequencial_samples(self, batch_size, start_idx):
        print(start_idx)
        return [self.buffer[ii] for ii in range(start_idx, start_idx + batch_size)]

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

    def get_action(self, state, experienced_episodes, mainQN, isBacktest = False):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        # TODO: 学習進捗をみて式の調整をしていく必要あり
        epsilon = 0.001 + 0.9 / (1.0 + (300.0 * (experienced_episodes / TOTAL_ACTION_NUM)))

        # epsilonが小さい値の場合の方が最大報酬の行動が起こる
        # イテレーション数が5の倍数の時か、バックテストの場合は常に最大報酬の行動を選ぶ
        if epsilon <= np.random.uniform(0, 1) or isBacktest == True:
            reshaped_state = np.reshape(state, [1, time_series, feature_num])
            retTargetQs = mainQN.model.predict(reshaped_state)
            print("NN all output at get_action: " + str(list(itertools.chain.from_iterable(retTargetQs))))
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
            action = action - 1 # 0, 1, 2 を -1, 0, 1 に置き換える
        else:
            # ランダムに行動する
            action = np.random.choice([-1, 0, 1])

        return action

# ---
hidden_size_lstm1 = 64 #32
hidden_size_lstm2 = 32
hidden_size_dense1 = 64
hidden_size_dense2 = 32

learning_rate = 0.0001 #0.0016
time_series = 64 #32
batch_size = 64 #256 #1024
TRAIN_DATA_NUM = 252 * 5 # 5year #72000
num_episodes = TRAIN_DATA_NUM + 10  # envがdoneを返すはずなので念のため多めに設定
iteration_num = 5000 #720
memory_size = TRAIN_DATA_NUM * 2 + 10
feature_num = 3
nn_output_size = 3
TOTAL_ACTION_NUM = TRAIN_DATA_NUM * iteration_num
HODABLE_POSITIONS = 1 #30
BACKTEST_ITR_PERIOD = 30
half_spread = 0.0015

gamma = 0.3
volatility_tgt = 3.0 #2.0
bp = 0.000015 # 1ドル100円の時にスプレッドで0.15銭とられるよう逆算した比率

train_episode_interval = 1024 # bs64 * 16

SELL = -1 #SELL
DONOT = 0 #DONOT
BUY = 1   #BUY

def tarin_agent():
    env_master = FXEnvironment(TRAIN_DATA_NUM, time_series=time_series, holdable_positions=HODABLE_POSITIONS, half_spread=half_spread, volatility_tgt=volatility_tgt, bp=bp)
    mainQN = QNetwork(time_series=time_series, learning_rate=learning_rate, state_size=feature_num, action_size=nn_output_size)     # メインのQネットワーク
    targetQN = QNetwork(time_series=time_series, learning_rate=learning_rate, state_size=feature_num, action_size=nn_output_size)     # Double DQNのためのネットワーク
    targetQNtmp = QNetwork(time_series=time_series, learning_rate=learning_rate, state_size=feature_num, action_size=nn_output_size)     # Double DQN実現のための一時変数

    memory = Memory([], max_size=memory_size)
    actor = Actor()

    total_get_action_cnt = 1

    # if os.path.exists("./mainQN.hd5"):
    #     mainQN.load_model("mainQN")
    #     # memory.load_memory("memory")
    #     # with open("./total_get_action_count.pickle", 'rb') as f:
    #     #     total_get_acton_cnt = pickle.load(f)
    #     # with open("./all_period_reward_arr.pickle", 'rb') as f:
    #     #     all_period_reward_arr = pickle.load(f)

    def store_episode_log_to_memory(state, action, reward, next_state):
        nonlocal memory
        # nonlocal state_x_action_hash
        a_log = [state, action, reward, next_state]
        memory.add(a_log)  # メモリを更新する

    #######################################################

    for cur_itr in range(iteration_num):
        # 定期的にバックテストを行い評価できるようにしておく（CSVを吐く）
        if cur_itr % BACKTEST_ITR_PERIOD == 0 and cur_itr != 0:
            mainQN.save_model("mainQN")
            run_backtest("auto_backtest", env_master=env_master)
            run_backtest("auto_backtest_test", env_master=env_master)
            continue

        env = env_master.get_env('train')
        #action = np.random.choice([0, 1, 2])
        action = np.random.choice([-1, 0, 1])
        state, reward, done = env.step(action)  # 1step目は適当なBUYかDONOTのうちランダムに行動をとる
        total_get_action_cnt += 1
        state = np.reshape(state, [time_series, feature_num])  # list型のstateを、1行15列の行列に変換
        # ここだけ 同じstateから同じstateに遷移したことにする
        store_episode_log_to_memory(state, action, reward, state)

        # Double DQNを実現するためにテンポラリなネットワークも挟んで前イテレーションのネットワークを
        # 利用できるようにしておく
        targetQN.model.set_weights(targetQNtmp.model.get_weights())
        targetQNtmp.model.set_weights(mainQN.model.get_weights())

        for episode in range(num_episodes):  # 試行数分繰り返す
            total_get_action_cnt += 1

            # 時刻tでの行動を決定する
            action = actor.get_action(state, total_get_action_cnt, mainQN)

            next_state, reward, done = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する

            # 環境が提供する期間が最後までいった場合
            if done:
                print(str(cur_itr) + ' training period finished.')
                break

            next_state = np.reshape(next_state, [time_series, feature_num])  # list型のstateを、1行feature num列の行列に変換
            store_episode_log_to_memory(state, action, reward, next_state)

            state = next_state  # 状態更新

            # 1024エピソードごとにfitする（それでFixed DQNを実装したことになるはず）
            # Qネットワークの重みを学習・更新する replay
            # train_episode_interval分新たにmemoryにエピソードがたまったら batch_size のミニバッチ 16個 として replayする
            if episode + 1 >= train_episode_interval and (episode + 1) % train_episode_interval == 0:
                mainQN.replay(memory, time_series, targetQN, cur_episode_idx=episode, batch_num=16)

        # 次周では過去のmemoryは参照しない
        memory.clear()

def run_backtest(backtest_type, env_master=None):
    if env_master:
        env_master_local = env_master
    else:
        env_master_local = FXEnvironment(TRAIN_DATA_NUM, time_series=time_series, holdable_positions=HODABLE_POSITIONS, half_spread=half_spread, volatility_tgt=volatility_tgt, bp=bp)

    env = env_master_local.get_env(backtest_type)
    num_episodes = 1500000  # 10年. envがdoneを返すはずなので適当にでかい数字を設定しておく

    mainQN = QNetwork(learning_rate=learning_rate, time_series=time_series)     # メインのQネットワーク
    actor = Actor()

    mainQN.load_model("mainQN")

    # DONOT でスタート
    state, reward, done = env.step(0)
    state = np.reshape(state, [time_series, feature_num])
    for episode in range(num_episodes):   # 試行数分繰り返す
        action = actor.get_action(state, episode, mainQN, isBacktest = True)   # 時刻tでの行動を決定する

        state, reward, done  = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
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

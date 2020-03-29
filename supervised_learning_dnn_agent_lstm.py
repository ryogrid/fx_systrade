# coding:utf-8
# [0]必要なライブラリのインポート

# this code based on code on https://qiita.com/sugulu/items/bc7c70e6658f204f85f9
# I am very grateful to work of Mr. Yutaro Ogawa (id: sugulu)

USE_TENSOR_BOARD = True

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json, Model, load_model, save_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LSTM, RepeatVector, TimeDistributed, Reshape, LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l1, l2
from collections import deque
import pickle
from supervised_learning_dnn_environment_lstm import FXEnvironment
import os
import sys
import random
import itertools
import math

# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.001, state_size=15, action_size=3, time_series=32):
        self.optimizer = Adam(lr=learning_rate, clipvalue=0.5)
        #self.optimizer = RMSprop(lr=learning_rate, momentum=0.9, clipvalue=0.1)
        #self.optimizer = SGD(lr=learning_rate, momentum=0.9, clipvalue=0.5)

        self.loss_func = tf.keras.losses.Huber(delta=1.0)

        self.model = tf.keras.Sequential([
            LSTM(hidden_size, input_shape=(time_series, state_size), return_sequences=True, activation=None, kernel_regularizer=l1(0.1)), #recurrent_dropout=0.5),
            LeakyReLU(0.2),
            #PReLU(),
            BatchNormalization(),
            Dropout(0.5),
            LSTM(hidden_size, return_sequences=False, activation=None, kernel_regularizer=l1(0.1)), #recurrent_dropout=0.5),
            LeakyReLU(0.2),
            #PReLU(),
            BatchNormalization(),
            Dense(action_size, activation='softmax')
        ])
        #self.model.compile(optimizer=self.optimizer, loss=self.loss_func)
        #self.model.compile(optimizer=self.optimizer, loss="sparse_categorical_crossentropy", metrics = ['accuracy'])
        self.model.compile(optimizer=self.optimizer, loss=self.loss_func, metrics=['accuracy'])
        self.model.summary()


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

class Actor:
    def __init__(self):
        pass

    def get_action(self, features, mainQN):
            # epsilonが小さい値の場合の方が最大報酬の行動が起こる
            # イテレーション数が5の倍数の時か、バックテストの場合は常に最大報酬の行動を選ぶ
            reshaped_state = np.reshape(features, [1, time_series, feature_num])
            retTargetQs = mainQN.model.predict(reshaped_state)
            print("NN all output at get_action: " + str(list(itertools.chain.from_iterable(retTargetQs))))
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する

            return action

    def train(self, mainNN, train_x, train_y, validation_x, validation_y):
        batch_num = len(train_x) // batch_size
        validation_len = len(validation_x)


        train_x = np.array(train_x[:batch_size*batch_num])
        train_y = np.array(train_y[:batch_size*batch_num])
        validation_x = np.array(validation_x)
        validation_y = np.array(validation_y)

        train_x = train_x.reshape((batch_size*batch_num, time_series, feature_num))
        train_y = train_y.reshape((batch_size*batch_num, nn_output_size))
        validation_x = validation_x.reshape((validation_len, time_series, feature_num))
        validation_y = validation_y.reshape((validation_len, nn_output_size))

        print(train_x.shape)
        print(train_y.shape)

        cbks = []
        if USE_TENSOR_BOARD:
            # tensorboardのためのデータを記録するコールバック
            callbacks = tf.keras.callbacks.TensorBoard(log_dir='logdir', histogram_freq=1,
                                                    write_graph=True, write_grads=True, profile_batch=True)
            cbks = [callbacks]

        #ベストモデルを自動保存するようコールバックを設定
        snapshot_cbk = tf.keras.callbacks.ModelCheckpoint(
            "./best_model", monitor='val_accuracy', verbose=1, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch'
        )
        cbks.append(snapshot_cbk)

        mainNN.model.fit(x=train_x, y=train_y, validation_data=(validation_x, validation_y), validation_freq = 4, epochs=epochs,
                         verbose=1, batch_size=batch_size, callbacks=cbks)


hidden_size = 64
learning_rate = 0.0004
time_series = 64
batch_size = 256
TRAIN_DATA_NUM = 72000 # <- 5分足で1年 # 36000 - time_series # <- 10分足で1年
num_episodes = TRAIN_DATA_NUM + 10  # envがdoneを返すはずなので念のため多めに設定 #1000  # 総試行回数
feature_num = 10
nn_output_size = 2 #3
HODABLE_POSITIONS = 100 #30
predict_future_legs = 40
epochs = 4000 #45 #15 #45 # 90 #400
half_spread = 0.0015

BUY = 0
SELL = 1
DONOT = 2
CLOSE = 3

def tarin_agent():
    env_master = FXEnvironment(train_data_num = TRAIN_DATA_NUM, time_series=time_series, holdable_positions=HODABLE_POSITIONS, predict_future_legs=40, half_spread = half_spread)
    mainQN = QNetwork(time_series=time_series, learning_rate=learning_rate, state_size=feature_num, action_size=nn_output_size)     # メインのQネットワーク
    actor = Actor()

    # inputは time_series個ずつ まとまった形で返ってくる
    tr_input_arr, tr_label_arr, ts_input_arr, ts_label_arr = env_master.get_train_and_validation_datas()

    actor.train(mainQN, tr_input_arr, tr_label_arr, ts_input_arr, ts_label_arr)
    mainQN.save_model("mainQN")

def run_backtest(backtest_type, learingQN=None):
    close_position_episode_idx_arr = [[-1] for i in range(TRAIN_DATA_NUM)]

    env_master = FXEnvironment(time_series=time_series, holdable_positions=HODABLE_POSITIONS, half_spread=0.0015)
    env = env_master.get_env(backtest_type)
    num_episodes = 1500000  # 10年. envがdoneを返すはずなので適当にでかい数字を設定しておく

    # [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    mainQN = QNetwork(learning_rate=learning_rate, time_series=time_series)     # メインのQネットワーク
    actor = Actor()

    mainQN.load_model("mainQN")

    # DONOT でスタート
    state, reward, done, info, needclose = env.step(0)
    state = np.reshape(state, [time_series, feature_num])
    for episode in range(num_episodes):   # 試行数分繰り返す
        if close_position_episode_idx_arr[episode] != -1:
            action = CLOSE
        else:
            action = actor.get_action(state, episode, mainQN)   # 時刻tでの行動を決定する
            if action == BUY or action == SELL:
                close_position_episode_idx_arr[episode + predict_future_legs] = 1

        state, done = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
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
    np.random.seed(1337)  # for reproducibility

    # バックテストだけ行う際はGPUで predictすると遅いので搭載されてないものとして動作させる
    if sys.argv[1] == "train":
        disable_gpu()
        #limit_gpu_memory_usage()
        tarin_agent()
    elif sys.argv[1] == "backtest":
        disable_gpu()
        run_backtest("backtest")
    elif sys.argv[1] == "backtest_test":
        disable_gpu()
        run_backtest("backtest_test")
    else:
        print("please pass argument 'train' or 'backtest'")

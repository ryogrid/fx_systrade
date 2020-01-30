# coding:utf-8
# [0]必要なライブラリのインポート

# this code based on code on https://qiita.com/sugulu/items/bc7c70e6658f204f85f9
# I am very grateful to work of Mr. Yutaro Ogawa (id: sugulu)

import numpy as np
import time
from keras.models import Sequential, model_from_json
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from keras import backend as K
import tensorflow as tf
import pickle
from agent_fx_environment import FXEnvironment
import os
import sys
import math

# [1]損失関数の定義
# 損失関数にhuber関数を使用 参考https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
#def huberloss(y_true, y_pred):
#    err = y_true - y_pred
#    cond = K.abs(err) < 1.0
#    L2 = 0.5 * K.square(err)
#    L1 = (K.abs(err) - 0.5)
#    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
#    return K.mean(loss)

reward_arr = np.array([])
cur_idx_for_loss_calc = -1
mu = 1.0
sigma = 0.04
SRATIO_PERIOD = 64

def huberloss(y_true, y_pred):
    return tf.compat.v1.losses.huber_loss(y_true, y_pred)

def sharpratio_loss(y, x):
    global cur_idx_for_loss_calc

    partial_reward_mean = np.mean(mu*(reward_arr[cur_idx_for_loss_calc - SRATIO_PERIOD:cur_idx_for_loss_calc]))
    partial_reward_std = np.std(reward_arr[cur_idx_for_loss_calc - SRATIO_PERIOD:cur_idx_for_loss_calc])
    cur_idx_for_loss_calc += 1
    return partial_reward_std / partial_reward_mean


# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.001, state_size=15, action_size=3, hidden_size=10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(BatchNormalization())
        #self.model.add(Dropout(0.5))

        self.model.add(Dense(action_size, activation='tanh'))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        #self.optimizer = Adam()  # 誤差を減らす学習方法はAdam. 学習係数はAdam optimizerのデフォルト値を使う.
        # self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)


    # # 重みの学習
    # def replay(self, memory, batch_size, gamma):
    #     inputs = np.zeros((batch_size, feature_num))
    #     targets = np.zeros((batch_size, 3))
    #     mini_batch = memory.sample(batch_size)
    #
    #     for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
    #         #inputs[i:i + 1] = state_b
    #         inputs[i] = state_b
    #         target = reward_b
    #
    #         retmainQs = self.model.predict(next_state_b)[0]
    #         print(retmainQs)
    #         next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
    #         target = reward_b + gamma * retmainQs[next_action]
    #
    #
    #         targets[i] = self.model.predict(state_b)    # Qネットワークの出力
    #         targets[i][action_b] = target               # 教師信号
    #
    #     self.model.fit(inputs, targets, epochs=1, verbose=1)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定

    # # 重みの学習
    # def replay(self, memory, batch_size, gamma):
    #     inputs = np.zeros((batch_size, feature_num))
    #     targets = np.zeros((batch_size, 3))
    #     mini_batch = memory.sample(batch_size)
    #
    #     for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
    #         inputs[i:i + 1] = state_b
    #
    #         targets[i] = self.model.predict(state_b)  # Qネットワークの出力
    #         targets[i][action_b] = reward_b  # 教師信号
    #
    #     self.model.fit(inputs, targets, epochs=1, verbose=1)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定

    def save_model(self, file_path_prefix_str):
        with open("./" + file_path_prefix_str + "_nw.json", "w") as f:
            f.write(self.model.to_json())
        self.model.save_weights("./" + file_path_prefix_str + "_weights.hd5")

    def load_model(self, file_path_prefix_str):
        with open("./" + file_path_prefix_str + "_nw.json", "r") as f:
            self.model = model_from_json(f.read())
        self.model.compile(loss=huberloss, optimizer=self.optimizer)
        self.model.load_weights("./" + file_path_prefix_str + "_weights.hd5")

# [3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

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

    def len(self):
        return len(self.buffer)

    def save_memory(self, file_path_prefix_str):
        with open("./" + file_path_prefix_str + ".pickle", 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_memory(self, file_path_prefix_str):
        with open("./" + file_path_prefix_str + ".pickle", 'rb') as f:
            self.buffer = pickle.load(f)

# [4]カートの状態に応じて、行動を決定するクラス
class Actor:
    def get_action(self, state, episode, mainQN, isBacktest = False):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0+(episode/(iteration_num * 100)))

        if epsilon <= np.random.uniform(0, 1) or isBacktest == True:
            retTargetQs = mainQN.model.predict(state)[0]
            #print(retTargetQs)
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
            action = round(action)
        else:
            action = np.random.choice([0, 1, 2])  # ランダムに行動する

        return action

# [5] メイン関数開始----------------------------------------------------
# [5.1] 初期設定--------------------------------------------------------
DQN_MODE = 1    # 1がDQN、0がDDQNです
TRAIN_DATA_NUM = 223954 # 3years (test is 5 years)
# ---
gamma = 0.99  # 割引係数
hidden_size = 50 #100 #50  # 16               # Q-networkの隠れ層のニューロンの数
learning_rate = 0.005 #0.01 #0.001 #0.0001 # 0.00001         # Q-networkの学習係数
memory_size = 7000000 #10000  # バッファーメモリの大きさ
batch_size = 64 #32  # Q-networkを更新するバッチの大きさ
num_episodes = TRAIN_DATA_NUM + 10  # envがdoneを返すはずなので念のため多めに設定 #1000  # 総試行回数
iteration_num = 160 #25
feature_num = 13 #10 # 11
nn_output_size = 1

def tarin_agent():
    global reward_arr

    env_master = FXEnvironment()
    islearned = 0  # 学習が終わったフラグ

    # [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate, state_size=feature_num, action_size=nn_output_size)     # メインのQネットワーク
    # targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate, state_size=feature_num)   # 価値を計算するQネットワーク
    # plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
    memory = Memory(max_size=memory_size)
    actor = Actor()

    if os.path.exists("./mainQN_nw.json"):
        # 期間は最初からになってしまうが学習済みのモデルに追加で学習を行う
        mainQN.load_model("mainQN")
        # targetQN.load_model("targetQN")
        memory.load_memory("memory")

    total_get_acton_cnt = 1
    do_fit_count = 0

    inputs = np.zeros((batch_size, feature_num))
    targets = np.zeros((batch_size, nn_output_size))
    for cur_itr in range(iteration_num):
        env = env_master.get_env('train')
        state, reward, done = env.step(0)  # 1step目は適当な行動をとる ("HOLD")
        state = np.reshape(state, [1, feature_num])  # list型のstateを、1行15列の行列に変換

        for episode in range(num_episodes):  # 試行数分繰り返す
            # # 行動決定と価値計算のQネットワークをおなじにする
            # targetQN.model.set_weights(mainQN.model.get_weights())

            total_get_acton_cnt += 1
            action = actor.get_action(state, total_get_acton_cnt, mainQN)  # 時刻tでの行動を決定する
            #action = actor.get_action(state, episode, mainQN)   # 時刻tでの行動を決定する
            next_state, reward, done = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
            next_state = np.reshape(state, [1, feature_num])  # list型のstateを、1行11列の行列に変換
            np.insert(reward_arr, reward_arr.size, reward)

            memory.add((state, action, reward, next_state))     # メモリを更新する
            state = next_state  # 状態更新

            do_fit_count+= 1
            if do_fit_count % batch_size == 0 and do_fit_count != 0:
                mini_batch = memory.get_last(batch_size)

                for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
                    inputs[i] = state_b
                    target = reward_b

                    retmainQs = mainQN.model.predict(next_state_b)[0]
                    print(retmainQs)
                    #next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                    next_action_val = retmainQs[0]
                    target = reward_b + gamma * next_action_val

                    #targets[i] = mainQN.model.predict(state_b)  # Qネットワークの出力
                    targets[i][0] = target  # 教師信号

                cur_idx_for_loss_calc = episode
                mainQN.model.fit(inputs, targets, epochs=1, verbose=1, batch_size=batch_size)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
                inputs = np.zeros((batch_size, feature_num))
                targets = np.zeros((batch_size, nn_output_size))

            # if DQN_MODE:
            #     # 行動決定と価値計算のQネットワークをおなじにする
            #     targetQN.model.set_weights(mainQN.model.get_weights())

            # 環境が提供する期間が最後までいった場合
            if done:
                print('all training period learned.')
                break

            # モデルとメモリのスナップショットをとっておく
            if(episode % 10000 == 0):
                # targetQN.save_model("targetQN")
                mainQN.save_model("mainQN")
                memory.save_memory("memory")

def run_backtest(period_kind):
    env_master = FXEnvironment()
    env = env_master.get_env(period_kind)
    num_episodes = TRAIN_DATA_NUM + 10 # envがdoneを返すはずなので念のため多めに設定 #1000  # 総試行回数
    islearned = 0  # 学習が終わったフラグ

    # [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)     # メインのQネットワーク
    targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)   # 価値を計算するQネットワーク
    # plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
    actor = Actor()

    mainQN.load_model("mainQN")

    # HOLD でスタート
    state, reward, done = env.step(0)
    state = np.reshape(state, [1, feature_num])
    for episode in range(num_episodes):   # 試行数分繰り返す
        action = actor.get_action(state, episode, mainQN, isBacktest = True)   # 時刻tでの行動を決定する
        state, reward, done = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
        state = np.reshape(state, [1, feature_num])
        # 環境が提供する期間が最後までいった場合
        if done:
            print('all training period learned.')
            break

if __name__ == '__main__':
    if sys.argv[1] == "train":
        tarin_agent()
    elif sys.argv[1] == "backtest":
        run_backtest('train')
    else:
        print("please pass argument 'train' or 'backtest'")

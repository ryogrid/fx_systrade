# coding:utf-8
# [0]必要なライブラリのインポート

# this code based on code on https://qiita.com/sugulu/items/bc7c70e6658f204f85f9
# I am very grateful to work of Mr. Yutaro Ogawa (id: sugulu)

import numpy as np
import time
from keras.models import Sequential, model_from_json, Model
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
def huberloss(y_true, y_pred):
   err = y_true - y_pred
   cond = K.abs(err) < 1.0
   L2 = 0.5 * K.square(err)
   L1 = (K.abs(err) - 0.5)
   loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
   return K.mean(loss)

#reward_arr = np.array([])
#dummy_reward_arr = np.array([])
#mu = 1.0
#sigma = 0.04
#SRATIO_PERIOD = 64
#cur_fit_idx = [0]

# def huberloss(y_true, y_pred):
#     return tf.compat.v1.losses.huber_loss(y_true, y_pred)

# def dummyloss(y, x):
#     return y
#
# def sharpratio_loss_wrapper(r_arr, dummy_r_arr, idx_arr, period, mu_local):
#     def sharpratio_loss(y, x):
#         y_casted = tf.keras.backend.cast(y, 'float64')
#         x_casted = tf.keras.backend.cast(x, 'float64')
#         int_idx = int(idx_arr[0])
#         sharp_ratio = np.std(r_arr[int_idx - 2*period : int_idx - period] - dummy_r_arr[int_idx - 2*period : int_idx - period]) / \
#                                   (np.mean(r_arr[int_idx - 2*period : int_idx - period] - dummy_r_arr[int_idx - 2*period : int_idx - period]) + 0.00001)
#         print(sharp_ratio)
#         sharp_ratio = K.square(x_casted) * sharp_ratio
#         idx_arr[0] = idx_arr[0] + 1
#
#         return dummyloss(y_casted, x_casted) + mu_local * sharp_ratio #0.5 * K.square(sharp_ratio)
#
#     return sharpratio_loss

# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.001, state_size=15, action_size=3, hidden_size=10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))

        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        #self.model.compile(loss=sharpratio_loss_wrapper(reward_arr, dummy_reward_arr, cur_fit_idx, SRATIO_PERIOD, mu), optimizer=self.optimizer)
        self.model.compile(loss=huberloss,
                           optimizer=self.optimizer)


    # 重みの学習
    def replay(self, memory, batch_size, gamma):
        inputs = np.zeros((batch_size, feature_num))
        targets = np.zeros((batch_size, 2))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            #inputs[i:i + 1] = state_b
            inputs[i] = state_b

            # retmainQs = self.model.predict(next_state_b)[0]
            # print(retmainQs)
            # next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
            # target = reward_b + gamma * retmainQs[next_action]
            #
            # targets[i] = self.model.predict(state_b)    # Qネットワークの出力
            # targets[i][action_b] = target               # 教師信号

            another_action = 1 - action_b # action_bが 0 or 1 だとこの式で他方のインデックスが求まる
            another_val = -1 * reward_b # BUY or SELL で rewardは 1 or -1 なので他方は必ず符号を逆転した値になる

            # 以下だとただの教師あり学習! だが、あえてそうしている
            #targets[i] = self.model.predict(state_b)      # Qネットワークの出力
            targets[i][action_b] = reward_b               # 教師信号
            targets[i][another_action] = another_val      # もう一方のアクションのreward

        self.model.fit(inputs, targets, epochs=5, verbose=1)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定

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
            print(retTargetQs)
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
            # action_val = retTargetQs[0]
            # if math.isnan(action_val):
            #     action_val = 1
            #print(action_val)
            # action = round(action_val) + 1
        else:
            action = np.random.choice([0, 1])  # ランダムに行動する
            #action = np.random.choice([0, 1, 2])  # ランダムに行動する

        return action

# [5] メイン関数開始----------------------------------------------------
# [5.1] 初期設定--------------------------------------------------------
TRAIN_DATA_NUM = 223954 # 3years (test is 5 years)
# ---
gamma = 0.99  # 割引係数
hidden_size = 50 #100 #50  # 16               # Q-networkの隠れ層のニューロンの数
learning_rate = 0.01 # 0.05 #0.001 #0.0001 # 0.00001         # Q-networkの学習係数
memory_size = 1500000 #10000  # バッファーメモリの大きさ
batch_size = 32 #64 # 32  # Q-networkを更新するバッチの大きさ
num_episodes = TRAIN_DATA_NUM + 10  # envがdoneを返すはずなので念のため多めに設定 #1000  # 総試行回数
iteration_num = 7 # <- batch_size * replayでのepoch と掛け合わせて1000ぐらいになる #32 # #25
feature_num = 10 # 11
nn_output_size = 2 #3

def tarin_agent():
    #global reward_arr
    #global dummy_reward_arr
    #global cur_fit_idx

    env_master = FXEnvironment()

    # [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate, state_size=feature_num, action_size=nn_output_size)     # メインのQネットワーク
    memory = Memory(max_size=memory_size)
    actor = Actor()

    if os.path.exists("./mainQN_nw.json"):
        # 期間は最初からになってしまうが学習済みのモデルに追加で学習を行う
        mainQN.load_model("mainQN")
        # targetQN.load_model("targetQN")
        memory.load_memory("memory")

    total_get_acton_cnt = 1
    #do_fit_count = -1

    inputs = np.zeros((batch_size, feature_num))
    targets = np.zeros((batch_size, nn_output_size))
    for cur_itr in range(iteration_num):
        env = env_master.get_env('train')
        state, reward, done = env.step(0)  # 1step目は適当な行動をとる ("BUY")
        state = np.reshape(state, [1, feature_num])  # list型のstateを、1行15列の行列に変換

        for episode in range(num_episodes):  # 試行数分繰り返す
            # # 行動決定と価値計算のQネットワークをおなじにする
            # targetQN.model.set_weights(mainQN.model.get_weights())

            total_get_acton_cnt += 1
            action = actor.get_action(state, total_get_acton_cnt, mainQN)  # 時刻tでの行動を決定する
            next_state, reward, done = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
            next_state = np.reshape(state, [1, feature_num])  # list型のstateを、1行11列の行列に変換
            # reward_arr = np.insert(reward_arr, reward_arr.size, reward)
            # dummy_reward_arr = np.insert(dummy_reward_arr, dummy_reward_arr.size, 0.0)

            memory.add((state, action, reward, next_state))     # メモリを更新する
            state = next_state  # 状態更新

            # Qネットワークの重みを学習・更新する replay
            if (memory.len() > batch_size):
                mainQN.replay(memory, batch_size, gamma)

            # do_fit_count+= 1
            # if do_fit_count % batch_size == 0 and do_fit_count != 0 and reward_arr.size >= 2*SRATIO_PERIOD:
            #     mini_batch = memory.get_last(batch_size)
            #     for i, (state_b, action_b, reward_b) in enumerate(mini_batch):
            #         inputs[i] = state_b
            #         # loss関数は与えた教師信号をみないが、その後の重みの更新でいりそうなので
            #         # 予測した値を与えておく
            #         targets[i][0] = mainQN.model.predict(state_b)[0][0]
            #
            #     cur_fit_idx[0] = episode
            #     mainQN.model.fit(inputs, targets, epochs=1, verbose=1,
            #                      batch_size=batch_size)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
            #     inputs = np.zeros((batch_size, feature_num))
            #     targets = np.zeros((batch_size, nn_output_size))

            # 環境が提供する期間が最後までいった場合
            if done:
                print('all training period learned.')
                break

            # モデルとメモリのスナップショットをとっておく
            if episode % 10000 == 0 and episode != 0:
                # targetQN.save_model("targetQN")
                mainQN.save_model("mainQN")
                memory.save_memory("memory")

        # reward_arr = np.array([])
        # dummy_reward_arr = np.array([])
        # do_fit_count = -1


def run_backtest():
    env_master = FXEnvironment()
    env = env_master.get_env("backtest")
    num_episodes = TRAIN_DATA_NUM + 10 # envがdoneを返すはずなので念のため多めに設定 #1000  # 総試行回数

    # [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)     # メインのQネットワーク
    targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)   # 価値を計算するQネットワーク
    # plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
    actor = Actor()

    mainQN.load_model("mainQN")

    # BUY でスタート
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
    np.random.seed(1337)  # for reproducibility
    if sys.argv[1] == "train":
        tarin_agent()
    elif sys.argv[1] == "backtest":
        run_backtest()
    else:
        print("please pass argument 'train' or 'backtest'")

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
import gym

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
    def __init__(self, learning_rate=0.001, state_size=15, action_size=3, hidden_size=10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))

        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self.model.compile(loss=huberloss,
                           optimizer=self.optimizer)

    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQNarg = None):
        inputs = np.zeros((batch_size, feature_num))
        targets = np.zeros((batch_size, 3))
        mini_batch = memory.sample(batch_size)
        targetQN = targetQNarg
        if targetQNarg == None:
            targetQN = self.model

        #mini_batch = memory.get_last(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i+1] = state_b

            retmainQs = self.model.predict(next_state_b)[0]
            next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
            #target = reward_b + gamma * retmainQs[next_action]
            next_state_max_reward = targetQN.model.predict(next_state_b)[0][next_action]
            target = reward_b + gamma * next_state_max_reward

            # # 以下はQ関数のマルコフ連鎖を考慮した更新式を無視した実装
            # # BUYとSELLのrewardが後追いで定まるため、それを反映するために replay を行う
            # targets[i] = self.model.predict(state_b)[0]    # Qネットワークの出力
            # # BUY or SELL で暫定の rewardとして 0 を返されている場合は、それで学習するとまずいので
            # # predictした結果を採用させる
            # if not ((action_b == 0 or action_b == 1) and reward_b == 0):

            targets[i] = self.model.predict(state_b)
            # BUYで暫定の rewardとして 0 を返されている場合は、それを用いて学習するとまずいので
            # predictした結果を採用させる（つまり、その場合以外であれば target を教師信号とする）
            targets[i][action_b] = target  # 教師信号
            #targets[i][2] = 0.0 + gamma * next_state_max_reward  # 教師信号（DONOTで返されるrewardは常に0)

        self.model.fit(inputs, targets, epochs=1, verbose=1)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定

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

# [4]カートの状態に応じて、行動を決定するクラス
class Actor:
    def get_action(self, state, episode, mainQN, isBacktest = False):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0 + (300.0 * (episode/TOTAL_ACTION_NUM)))

        if epsilon <= np.random.uniform(0, 1) or isBacktest == True:
            retTargetQs = mainQN.model.predict(state)[0]
            #print(retTargetQs)
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
        else:
            action = np.random.choice([0, 1, 2])  # ランダムに行動する

        return action

# [5] メイン関数開始----------------------------------------------------
# [5.1] 初期設定--------------------------------------------------------
# ---
gamma = 0.99 #0.3 #0.99  # 割引係数
hidden_size = 20 # Q-networkの隠れ層のニューロンの数
learning_rate = 0.001 #0.005 #0.01 # 0.05 #0.001 #0.0001 # 0.00001         # Q-networkの学習係数
batch_size = 32 #64 # 32  # Q-networkを更新するバッチの大きさ
num_episodes = 201 # envがdoneを返すはずなので念のため多めに設定 #1000  # 総試行回数
iteration_num = 1000 # <- 1足あたり 32 * 1 * 50 で約1500回のfitが行われる計算 #20
memory_size = num_episodes * int(iteration_num * 0.1) #10000  # バッファーメモリの大きさ
feature_num = 2 #10 #11
nn_output_size = 3
TOTAL_ACTION_NUM = num_episodes * iteration_num

def train_agent():
    env = gym.make('MountainCar-v0')

    # [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate, state_size=feature_num, action_size=nn_output_size)     # メインのQネットワーク
    targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate, state_size=feature_num,
                      action_size=nn_output_size)  # 状態の価値を求めるためのネットワーク
    memory = Memory(max_size=memory_size)
    actor = Actor()

    for cur_itr in range(iteration_num):
        env.reset()
        state, reward, done, info = env.step(env.action_space.sample())  # 1step目は適当な行動をとる ("DONOT")
        state = np.reshape(state, [1, feature_num])  # list型のstateを、1行15列の行列に変換

        # 状態の価値を求めるネットワークに、行動を求めるメインのネットワークの重みをコピーする（同じものにする）
        targetQN.model.set_weights(mainQN.model.get_weights())

        for episode in range(num_episodes):  # 試行数分繰り返す
            action = actor.get_action(state, cur_itr, mainQN)  # 時刻tでの行動を決定する
            next_state, reward, done, info = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
            next_state = np.reshape(next_state, [1, feature_num])  # list型のstateを、1行11列の行列に変換

            reward=0
            if done:
                reward=state[0][0]
            a_log = [state, action, reward, next_state]
            memory.add(a_log)     # メモリを更新する

            state = next_state  # 状態更新

            # Qネットワークの重みを学習・更新する replay
            if (memory.len() > batch_size):
                mainQN.replay(memory, batch_size, gamma, targetQNarg=targetQN)

            # 環境が提供する期間が最後までいった場合
            if done:
                print(str(cur_itr)+":"+str(reward))
                break

if __name__ == '__main__':
    train_agent()

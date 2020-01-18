# coding:utf-8
# [0]必要なライブラリのインポート
import gym  # 倒立振子(cartpole)の実行環境
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from gym import wrappers  # gymの画像保存
from keras import backend as K
import tensorflow as tf


# [1]損失関数の定義
# 損失関数にhuber関数を使用します 参考https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)


# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        # self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)

    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, 4))
        targets = np.zeros((batch_size, 2))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)    # Qネットワークの出力
            targets[i][action_b] = target               # 教師信号

        # shiglayさんよりアドバイスいただき、for文の外へ修正しました
        self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定


# [3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)


# [4]カートの状態に応じて、行動を決定するクラス
# アドバイスいただき、引数にtargetQNを使用していたのをmainQNに修正しました
class Actor:
    def get_action(self, state, episode, mainQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0+episode)

        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = mainQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する

        else:
            action = np.random.choice([0, 1])  # ランダムに行動する

        return action


# [5] メイン関数開始----------------------------------------------------
# [5.1] 初期設定--------------------------------------------------------
DQN_MODE = 0    # 1がDQN、0がDDQNです
LENDER_MODE = 0 # 0は学習後も描画なし、1は学習終了後に描画する

env = gym.make('CartPole-v0')
num_episodes = 1000  # 総試行回数
max_number_of_steps = 200  # 1試行のstep数
goal_average_reward = 195  # この報酬を超えると学習終了
num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納
gamma = 0.99    # 割引係数
islearned = 0  # 学習が終わったフラグ
isrender = 0  # 描画フラグ
# ---
hidden_size = 16               # Q-networkの隠れ層のニューロンの数
learning_rate = 0.0001 #0.00001         # Q-networkの学習係数
memory_size = 10000            # バッファーメモリの大きさ
batch_size = 32                # Q-networkを更新するバッチの大記載

# [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)     # メインのQネットワーク
targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)   # 価値を計算するQネットワーク
# plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
memory = Memory(max_size=memory_size)
actor = Actor()

# [5.3]メインルーチン--------------------------------------------------------
for episode in range(num_episodes):  # 試行数分繰り返す
    env.reset()  # cartPoleの環境初期化
    state, reward, done, _ = env.step(env.action_space.sample())  # 1step目は適当な行動をとる
    state = np.reshape(state, [1, 4])   # list型のstateを、1行4列の行列に変換
    episode_reward = 0


    # 2018.05.16
    # skanmeraさんより間違いを修正いただきました
    # targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする
    # ↓
    targetQN.model.set_weights(mainQN.model.get_weights())

    for t in range(max_number_of_steps + 1):  # 1試行のループ
        if (islearned == 1) and LENDER_MODE:  # 学習終了したらcartPoleを描画する
            env.render()
            time.sleep(0.1)
            print(state[0, 0])  # カートのx位置を出力するならコメントはずす

        action = actor.get_action(state, episode, mainQN)   # 時刻tでの行動を決定する
        next_state, reward, done, info = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
        next_state = np.reshape(next_state, [1, 4])     # list型のstateを、1行4列の行列に変換

        # 報酬を設定し、与える
        if done:
            next_state = np.zeros(state.shape)  # 次の状態s_{t+1}はない
            if t < 195:
                reward = -1  # 報酬クリッピング、報酬は1, 0, -1に固定
            else:
                reward = 1  # 立ったまま195step超えて終了時は報酬
        else:
            reward = 0  # 各ステップで立ってたら報酬追加（はじめからrewardに1が入っているが、明示的に表す）

        episode_reward += 1 # reward  # 合計報酬を更新

        memory.add((state, action, reward, next_state))     # メモリの更新する
        state = next_state  # 状態更新


        # Qネットワークの重みを学習・更新する replay
        if (memory.len() > batch_size) and not islearned:
            mainQN.replay(memory, batch_size, gamma, targetQN)

        if DQN_MODE:
        # 2018.06.12
        # shiglayさんさんより間違いを修正いただきました
        # targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする
        # ↓
            targetQN.model.set_weights(mainQN.model.get_weights())

        # 1施行終了時の処理
        if done:
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # 報酬を記録
            print('%d Episode finished after %f time steps / episode_reward %f / mean of last %f episode %f' % (episode, t + 1, episode_reward, num_consecutive_iterations, total_reward_vec.mean()))
            break

    # 複数施行の平均報酬で終了を判断
    if total_reward_vec.mean() >= goal_average_reward:
        print('Episode %d train agent successfuly!' % episode)
        islearned = 1
        if isrender == 0:   # 学習済みフラグを更新
            isrender = 1

            # env = wrappers.Monitor(env, './movie/cartpoleDDQN')  # 動画保存する場合
            # 10エピソードだけでどんな挙動になるのか見たかったら、以下のコメントを外す
            # if episode>10:
            #    if isrender == 0:
            #        env = wrappers.Monitor(env, './movie/cartpole-experiment-1') #動画保存する場合
            #        isrender = 1
            #    islearned=1;
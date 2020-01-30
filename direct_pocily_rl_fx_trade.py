"""
REINFORCE(Policy Gradient)
"""

import collections
import numpy as np
import tensorflow as tf

from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
import pickle

import os
import math
import sys
import time

from agent_fx_environment import FXEnvironment

#env = gym.make('CartPole-v1')
#NUM_STATES = env.env.observation_space.shape[0]
#NUM_ACTIONS = env.env.action_space.n

NUM_STATES = 10
NUM_ACTIONS = 3
LEARNING_RATE = 0.0005

class PolicyEstimator():
    def __init__(self):
        l_input = Input(shape=(NUM_STATES,))
        l_dense = Dense(20, activation='relu')(l_input)
        action_probs = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        self.model = Model(inputs=[l_input], outputs=[action_probs])

        if os.path.exists("./policyEstimator_nw.json"):
            # 期間は最初からになってしまうが学習済みのモデルに追加で学習を行う
            self.load_model("policyEstimator")

        self.state, self.action, self.target, self.action_probs, self.minimize, self.loss = self._build_graph(self.model)

    def _build_graph(self, model):
        state = tf.placeholder(tf.float32)
        action = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        target = tf.placeholder(tf.float32, shape=(None))

        action_probs = model(state)
        log_prob = tf.log(tf.reduce_sum(action_probs * action))
        loss = -log_prob * target

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        minimize = optimizer.minimize(loss)

        return state, action, target, action_probs, minimize, loss

    def predict(self, sess, state):
        return sess.run(self.action_probs, {self.state: [state]})

    def update(self, sess, state, action, target):
        feed_dict = {self.state: [state], self.target: target, self.action: to_categorical(action, NUM_ACTIONS)}
        _, loss = sess.run([self.minimize, self.loss], feed_dict)
        return loss

    def save_model(self, file_path_prefix_str):
        with open("./" + file_path_prefix_str + "_nw.json", "w") as f:
            f.write(self.model.to_json())
        self.model.save_weights("./" + file_path_prefix_str + "_weights.hd5")

    def load_model(self, file_path_prefix_str):
        with open("./" + file_path_prefix_str + "_nw.json", "r") as f:
            self.model = model_from_json(f.read())
        #self.model.compile(loss=huberloss, optimizer=self.optimizer)
        self.model.load_weights("./" + file_path_prefix_str + "_weights.hd5")

MAXIMIZE_PERIOD = 64

def train(sess, policy_estimator, num_episodes, gamma=1.0):
    Step = collections.namedtuple("Step", ["state", "action", "reward"])
    env_master = FXEnvironment()

    for i_episode in range(1, num_episodes + 1):
        episode = []
        loss_list = []
        rewards = []
        partial_episode = []

        action_count = 0
        env = env_master.get_env('train')
        state, reward, done = env.step(0)  # 1step目は適当な行動をとる ("HOLD")
        while True:
            action_probs = policy_estimator.predict(sess, state)[0]
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            episode.append(Step(state=state, action=action, reward=reward))
            partial_episode.append(Step(state=state, action=action, reward=reward))

            if done:
                break

            state = next_state
            action_count += 1

            # モデルとメモリのスナップショットをとっておく
            if(action_count % 10000 == 0):
                policy_estimator.save_model("policyEstimator")

            if action_count % MAXIMIZE_PERIOD == 0 and action_count != 0:
                for t, step in enumerate(partial_episode):
                    partial_return = np.mean(rewards)
                    std_on_partial = np.std(rewards)
                    target = partial_return / std_on_partial # sharp ration on MAXIMIZE_PERIOD
                    loss = policy_estimator.update(sess, step.state, step.action, target)
                    loss_list.append(loss)
                partial_episode = []

        # log
        total_reward = sum(e.reward for e in episode)
        avg_loss = sum(loss_list) / len(loss_list)
        print('episode %s avg_loss %s reward: %d' % (i_episode, avg_loss, total_reward))

    return

def backtest(sess, policy_estimator, num_episodes, gamma=1.0):
    env_master = FXEnvironment()

    for i_episode in range(1, num_episodes + 1):
        env = env_master.get_env('train')
        state, reward, done = env.step(0)  # 1step目は適当な行動をとる ("HOLD")
        episode = []

        while True:
            action_probs = policy_estimator.predict(sess, state)[0]
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            if done:
                break

            state = next_state

        # log
        total_reward = sum(e.reward for e in episode)
        print('episode %s reward: %f last 100: %f' % (i_episode, total_reward))

def train_agent():
    policy_estimator = PolicyEstimator()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(sess, policy_estimator, 100000, gamma=1)

def run_backtest():
    policy_estimator = PolicyEstimator()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        backtest(sess, policy_estimator, 1, gamma=1)

if __name__ == '__main__':
    if sys.argv[1] == "train":
        train_agent()
    elif sys.argv[1] == "backtest":
        run_backtest('train')
    else:
        print("please pass argument 'train' or 'backtest'")
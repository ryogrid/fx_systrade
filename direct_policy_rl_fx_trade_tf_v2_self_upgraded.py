# coding:utf-8

"""
REINFORCE(Policy Gradient)
"""
import collections
import numpy as np
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers
#from keras import backend as K
import pickle

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import os
import math
import sys
import time

from agent_fx_environment import FXEnvironment

NUM_STATES = 10
NUM_ACTIONS = 3
LEARNING_RATE = 0.0005

#class MyModel(tf.keras.layers.Layer):
class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        #self.input_layer = layers.Input((10,))
        self.hidden1 = layers.Dense(20, activation="relu")
        self.hidden2 = layers.Dense(20, activation="relu")
        self.output_dense = layers.Dense(3, activation="softmax")

    def __call__(self, inputs):
        #x = self.input_layer(inputs)
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.output_dense(x)
        return x

# class PolicyEstimator():
#     def __init__(self):
#         l_input = Input(shape=(NUM_STATES,))
#         l_dense = Dense(20, activation='relu')(l_input)
#         action_probs = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
#         #self.model = Model(inputs=[l_input], outputs=[action_probs])
#         self.model = Model(inputs=l_input, outputs=action_probs)
#         if os.path.exists("./policyEstimator_nw.json"):
#             self.load_model("policyEstimator")
#         self.state, self.action, self.target, self.action_probs, self.minimize, self.loss = self._build_graph(self.model)
#
#     def _build_graph(self, model):
#
#         state = tf.keras.backend.placeholder(dtype='float32', shape=(None, NUM_STATES))
#         action = tf.keras.backend.placeholder(dtype='float32', shape=(None, NUM_ACTIONS))
#         target = tf.keras.backend.placeholder(dtype='float32', shape=(None))
#
#         action_probs = model(state)
#         log_prob = tf.math.log(tf.reduce_sum(input_tensor=action_probs * action))
#         loss = lambda: -1 * log_prob * target
#         print(log_prob)
#         print(target)
#
#         #optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
#         optimizer = Adam(LEARNING_RATE)
#         minimize = optimizer.minimize(loss, var_list=[log_prob, target])
#
#
#         return state, action, target, action_probs, minimize, loss
#
#     def predict(self, sess, state):
#         return sess.run(self.action_probs, {self.state: [state]})
#
#     def update(self, sess, state, action, target):
#         feed_dict = {self.state: [state], self.target: target, self.action: to_categorical(action, NUM_ACTIONS)}
#         _, loss = sess.run([self.minimize, self.loss], feed_dict)
#         print("loss: " + str(loss))
#         return loss
#
#     def save_model(self, file_path_prefix_str):
#         with open("./" + file_path_prefix_str + "_nw.json", "w") as f:
#             f.write(self.model.to_json())
#         self.model.save_weights("./" + file_path_prefix_str + "_weights.hd5")
#
#     def load_model(self, file_path_prefix_str):
#         with open("./" + file_path_prefix_str + "_nw.json", "r") as f:
#             self.model = model_from_json(f.read())
#         #self.model.compile(loss=huberloss, optimizer=self.optimizer)
#         self.model.load_weights("./" + file_path_prefix_str + "_weights.hd5")

MAXIMIZE_PERIOD = 64

#def train(sess, policy_estimator, num_episodes, gamma=1.0):
def train(num_episodes, gamma=1.0):
    tf.keras.backend.set_floatx('float64')

    Step = collections.namedtuple("Step", ["state", "action", "reward"])
    env_master = FXEnvironment()

    model = MyModel()
    optim = Adam(learning_rate = 0.005)

    def loss(pred, loss_arr, actions):
        print(pred)
        print(actions)
        log_prob = tf.math.log(tf.reduce_sum(input_tensor=pred[0][actions[0]] * tf.cast(actions, np.float64)[0]))
        return -1 * log_prob * loss_arr[0]

    @tf.function
    def train_on_batch(X, loss_arr, actions):
        with tf.GradientTape() as tape:
            pred = model(X) # Train mode
            loss_val = loss(pred, loss_arr, actions)
            # backward
            graidents = tape.gradient(loss_val, model.trainable_weights)
            # step optimizer
            optim.apply_gradients(zip(graidents, model.trainable_weights))
            # update accuracy
            #acc.update_state(y, pred)  # 評価関数に結果を足していく
            return loss_val

    for i_episode in range(1, num_episodes + 1):
        episode = []
        loss_list = []
        rewards = []
        #partial_episode = []

        action_count = 0
        env = env_master.get_env('train')
        state, reward, done = env.step(0)  # first step is HOLD
        while True:
            input_state = np.reshape(np.array(state), [1, 10])
            #action_probs = model(input_state)[0]
            action_probs = model(input_state)[0].numpy()
            #print(model(input_state))
            print(action_probs)
            # for ii, elem in enumerate(action_probs):
            #     print(elem)
            #     if elem < 0:
            #         action_probs[ii] = 0
            # sum = 0.0
            # for ii, elem in enumerate(action_probs):
            #     sum += elem

            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = env.step(action)
            rewards.append(reward)

            episode.append(Step(state=state, action=action, reward=reward))

            if done:
                break

            action_count += 1

            # # make snapshot
            # if(action_count % 10000 == 0):
            #     policy_estimator.save_model("policyEstimator")

            if action_count > MAXIMIZE_PERIOD:
                #for t, step in enumerate(partial_episode[[-1 * MAXIMIZE_PERIOD:]]):
                input = np.reshape(np.array(state), [1, 10])
                partial_return = np.mean(rewards[-1 * MAXIMIZE_PERIOD:])
                std_on_partial = np.std(rewards[-1 * MAXIMIZE_PERIOD:])
                target = partial_return / (std_on_partial + 0.00001) # sharp ration on MAXIMIZE_PERIOD
                if(math.isnan(target)):
                    target = 1.0
                    print("target is nan...")
                calculated_loss_arr = np.array([target])
                print(action)
                actions = np.array([action])
                loss_val = train_on_batch(input, calculated_loss_arr, actions)
                #loss = policy_estimator.update(sess, step.state, step.action, target)
                loss_list.append(loss)

            state = next_state

        # log
        total_reward = sum(e.reward for e in episode)
        avg_loss = sum(loss_list) / len(loss_list)
        print('episode %s avg_loss %s reward: %d' % (i_episode, avg_loss, total_reward))

    return

def backtest(sess, policy_estimator, num_episodes, gamma=1.0):
    env_master = FXEnvironment()

    for i_episode in range(1, num_episodes + 1):
        env = env_master.get_env('train')
        state, reward, done = env.step(0)  # 1step is HOLD
        episode = []

        while True:
            action_probs = policy_estimator.predict(sess, state)[0]
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = env.step(action)
            if done:
                break

            state = next_state

        # log
        total_reward = sum(e.reward for e in episode)
        print('episode %s reward: %f last 100: %f' % (i_episode, total_reward))

def train_agent():
    # policy_estimator = PolicyEstimator()
    # with tf.compat.v1.Session() as sess:
    #     sess.run(tf.compat.v1.global_variables_initializer())
    #     train(sess, policy_estimator, 100000, gamma=1)
    train(10000, gamma=1)

def run_backtest():
    #policy_estimator = PolicyEstimator()
    #with tf.compat.v1.Session() as sess:
    #    sess.run(tf.compat.v1.global_variables_initializer())
    #    backtest(sess, policy_estimator, 1, gamma=1)
    backtest(1, gamma=1)

if __name__ == '__main__':
    if sys.argv[1] == "train":
        train_agent()
    elif sys.argv[1] == "backtest":
        run_backtest('train')
    else:
        print("please pass argument 'train' or 'backtest'")
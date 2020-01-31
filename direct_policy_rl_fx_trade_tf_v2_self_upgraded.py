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
        self.hidden1 = layers.Dense(50, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros')
        self.bnormalize1 = layers.BatchNormalization()
        self.hidden2 = layers.Dense(50, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros')
        self.dropout1 = layers.Dropout(0.5)
        #self.output_dense = layers.Dense(1, activation="softmax")
        self.output_dense = layers.Dense(1, activation="tanh")

    def __call__(self, inputs):
        #x = self.input_layer(inputs)
        x = self.hidden1(inputs)
        x = self.bnormalize1(x)
        x = self.hidden2(x)
        x = self.dropout1(x)
        x = self.output_dense(x)
        return x



MAXIMIZE_PERIOD = 64

#def train(sess, policy_estimator, num_episodes, gamma=1.0):
def train(num_episodes, gamma=1.0):
    tf.keras.backend.set_floatx('float64')

    Step = collections.namedtuple("Step", ["state", "action", "reward"])
    env_master = FXEnvironment()

    model = MyModel()
    optim = Adam(learning_rate = 0.0001)

    #def loss(pred, loss_arr, actions):
    def loss(pred, loss_arr):
        #print(pred)
        #log_prob = tf.math.log(tf.reduce_sum(input_tensor=pred[0][actions[0]] * (tf.cast(actions, np.float64)[0]) + 100.0))
        log_prob = tf.math.log(
            tf.reduce_sum(input_tensor=pred[0][0] * 1.0 + 100.0))
        #return -1 * log_prob * loss_arr[0]
        return log_prob * loss_arr[0]

    @tf.function
    def train_on_batch(X, loss_arr):
    #def train_on_batch(X, loss_arr, actions):
        with tf.GradientTape() as tape:
            pred = model(X) # Train mode
            #loss_val = loss(pred, loss_arr, actions)
            loss_val = loss(pred, loss_arr)
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
            action = round(action_probs[0] + 1)
            #print(action)
            print(action_probs[0])
            #action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
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
                #target = partial_return / (std_on_partial + 0.00001) + 0.1 # sharp ration on MAXIMIZE_PERIOD
                target = std_on_partial / (partial_return + 0.00001) + 2.0  # sharp ration on MAXIMIZE_PERIOD
                #print(target)
                if(math.isnan(target)):
                    target = 1.0
                    print("target is nan...")
                calculated_loss_arr = np.array([target])
                # if action_count < 150000:
                #     action = np.random.randint(0, 3)
                # #print(action)
                # actions = np.array([action])
                # loss_val = train_on_batch(input, calculated_loss_arr, actions)
                loss_val = train_on_batch(input, calculated_loss_arr)
                print(loss_val)
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
    train(10000, gamma=1)

def run_backtest():
    backtest(1, gamma=1)

if __name__ == '__main__':
    if sys.argv[1] == "train":
        train_agent()
    elif sys.argv[1] == "backtest":
        run_backtest('train')
    else:
        print("please pass argument 'train' or 'backtest'")
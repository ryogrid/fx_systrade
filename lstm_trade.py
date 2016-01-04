import sys
import json
import numpy as np
import random
import pandas as pd
import pickle

import argparse
import math
import sys
import time
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F

mod = np
n_epoch   = 20 # # 39   # number of epochs
n_units   = 200 # 650  # number of units per layer
batchsize = 20   # minibatch size
bprop_len = 35   # length of truncated BPTT
grad_clip = 5    # gradient norm threshold to clip

INPUT_LEN = 20
OUTPUT_LEN = 20
TRAINING_DIV = 200

# Prepare RNNLM model
# model = FunctionSet(embed=F.EmbedID(INPUT_LEN, n_units),
#                     l1_x =F.Linear(n_units, 4 * n_units),
#                     l1_h =F.Linear(n_units, 4 * n_units),
#                     l2_x =F.Linear(n_units, 4 * n_units),
#                     l2_h =F.Linear(n_units, 4 * n_units),
#                     l3   =F.Linear(n_units, OUTPUT_LEN))
model = FunctionSet(embed=F.EmbedID(4, n_units),
                    l1_x =F.Linear(n_units, 4 * n_units),
                    l1_h =F.Linear(n_units, 4 * n_units),
                    l2_x =F.Linear(n_units, 4 * n_units),
                    l2_h =F.Linear(n_units, 4 * n_units),
                    l3   =F.Linear(n_units, 4))
for param in model.parameters:
    param[:] = np.random.uniform(-0.1, 0.1, param.shape)

# Neural net architecture
def forward_one_step(x_data, y_data, state, train=True):
    x = Variable(x_data, volatile=not train)
    t = Variable(y_data, volatile=not train)
    h0     = model.embed(x)
    h1_in  = model.l1_x(F.dropout(h0, train=train)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    h2_in  = model.l2_x(F.dropout(h1, train=train)) + model.l2_h(state['h2'])
    c2, h2 = F.lstm(state['c2'], h2_in)
    y      = model.l3(F.dropout(h2, train=train))
    state  = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
    if train == True:
        return state, F.softmax_cross_entropy(y, t)
    else:
        return state, F.softmax(y)

def make_initial_state(batchsize=batchsize, train=True):
    return {name: Variable(mod.zeros((batchsize, n_units), dtype=np.int32),
                           volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}

# Setup optimizer
optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model.collect_parameters())

# Evaluation routine
# def evaluate(dataset):
#     sum_log_perp = mod.zeros(())
#     state        = make_initial_state(batchsize=1, train=False)
#     for i in xrange(dataset.size - 1):
#         x_batch = dataset[i  :i+1]
#         y_batch = dataset[i+1:i+2]
#         state, loss   = forward_one_step(x_batch, y_batch, state, train=False)
#         sum_log_perp += loss.data.reshape(())
#     return math.exp(cuda.to_cpu(sum_log_perp) / (dataset.size - 1))

UP = 3
BIT_UP = 2
# EVEN = 2
BIT_DOWN = 1
DOWN = 0

"""
main
"""
rates_fd = open('./hoge.csv', 'r')
exchange_rates = []
prev = 0
for line in rates_fd:
    splited = line.split(",")
    if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>"and splited[0] != "204/04/26" and splited[0] != "20004/04/26":
        time_str = splited[0].replace("/", "-") + " " + splited[1]
        if prev == 0:
            val = BIT_UP
            prev = float(splited[2])
        else:
            diff = float(splited[2]) - prev
            prev = float(splited[2])
            if diff > 0.15:
                val = UP
            elif diff < -0.15:
                val = DOWN
            elif diff >= 0:
                val = BIT_UP
            elif diff < 0:
                val = BIT_DOWN
            else:
                val = BIT_UP # guard
                
        exchange_rates.append([time_str, val])

DATA_LEN = len(exchange_rates)

# Learning loop
train_len    = DATA_LEN / TRAINING_DIV
cur_log_perp = mod.zeros(())
epoch        = 0
start_at     = time.time()
cur_at       = start_at
state        = make_initial_state()
accum_loss   = Variable(mod.zeros(()))

print "data size: " + str(DATA_LEN)
print "train_len: " + str(train_len)
print "input len: " + str(INPUT_LEN)
print "output len: " + str(OUTPUT_LEN)
print "epoch num: " + str(n_epoch)

print 'going to train {} iterations'.format(train_len * n_epoch)
for epoch in xrange(n_epoch):
    for i in xrange(train_len):
        x_batch = np.array([exchange_rates[(INPUT_LEN + OUTPUT_LEN) * i + j][1]
                              # for j in xrange(batchsize)], dtype=np.int32)
                               for j in xrange(INPUT_LEN)], dtype=np.int32)
                            
        y_batch = np.array([exchange_rates[(INPUT_LEN + OUTPUT_LEN) * i + j + 1][1]
                              # for j in xrange(batchsize)], dtype=np.int32)
                               for j in xrange(OUTPUT_LEN)], dtype=np.int32)
        state, loss_i = forward_one_step(x_batch, y_batch, state)
        accum_loss   += loss_i
        cur_log_perp += loss_i.data.reshape(())

        if (epoch * train_len + i) % bprop_len == 0:  # Run truncated BPTT
            optimizer.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()  # truncate
            accum_loss = Variable(mod.zeros(()))

            optimizer.clip_grads(grad_clip)
            optimizer.update()

        if (epoch * train_len + i) % 1000 == 0:
            now      = time.time()
            throuput = 1000. / (now - cur_at)
            perp     = math.exp(cuda.to_cpu(cur_log_perp) / 1000)
            print 'iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(
                i + 1, perp, throuput)
            cur_at   = now
            cur_log_perp.fill(0)

        # if (i * epoch + 1) % jump == 0:
        if epoch * train_len + i % 100 == 0:            
            now  = time.time()
            # perp = evaluate(valid_data)
            # print 'epoch {} validation perplexity: {:.2f}'.format(epoch, perp)
            print 'epoch {}, i {}'.format(epoch, i)
            cur_at += time.time() - now  # skip time of evaluation

        if epoch >= 6:
            optimizer.lr /= 1.2
            print 'learning rate =', optimizer.lr

        sys.stdout.flush()

# Evaluate on test dataset
print 'test'
portfolio = 1000000
LONG = 1
SHORT = 2
NOT_HAVE = 3
pos_kind = NOT_HAVE
positions = 0

trade_val = -1

result = None
state = make_initial_state(batchsize=1, train=False)
for i in xrange(DATA_LEN - (DATA_LEN / TRAINING_DIV)):
    cur_idx = int((DATA_LEN / TRAINING_DIV) + i)
    x_batch = np.array([exchange_rates[cur_idx + j][1]
                           for j in xrange(INPUT_LEN)])
    # y_batch = np.array([exchange_rates[i + INPUT_LEN + j][1]
    #                        for j in xrange(OUTPUT_LEN)])
    state, loss_i = forward_one_step(x_batch, x_batch, state, train=False)

    print "last of input\n"
    print x_batch
    print "loss_i\n"
    print loss_i.data

    # print "state " + str(pos_kind)
    # print "predicted_angle " + str(predicted_angle)
    # if pos_kind == NOT_HAVE:
    #     if predicted_angle > 0:
    #         pos_kind = LONG
    #         positions = portfolio / exchange_rates[current_spot][1]
    #         trade_val = exchange_rates[current_spot][1]
    #     elif predicted_angle < 0:
    #         pos_kind = SHORT
    #         positions = portfolio / exchange_rates[current_spot][1]
    #         trade_val = exchange_rates[current_spot][1]
    # else:
    #     if pos_kind == LONG:
    #         pos_kind = NOT_HAVE
    #         portfolio = positions * exchange_rates[current_spot][1]
    #     elif pos_kind == SHORT:
    #         pos_kind = NOT_HAVE
    #         portfolio += positions * trade_val - positions * exchange_rates[current_spot][1]

    # print str(exchange_rates[current_spot][0]) + " " + str(portfolio)
    
    

    

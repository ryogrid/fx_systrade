import sys
import json
import numpy as np
import random
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.structure import RecurrentNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pandas as pd
import matplotlib.pyplot as plt

def construct_network(input_len, output_len, hidden_nodes, is_elman=True):
    n = RecurrentNetwork()
    n.addInputModule(LinearLayer(input_len, name="i"))
    n.addModule(BiasUnit("b"))
    n.addModule(SigmoidLayer(hidden_nodes, name="h"))
    n.addOutputModule(LinearLayer(output_len, name="o"))

    n.addConnection(FullConnection(n["i"], n["h"]))
    n.addConnection(FullConnection(n["b"], n["h"]))
    n.addConnection(FullConnection(n["b"], n["o"]))
    n.addConnection(FullConnection(n["h"], n["o"]))

    if is_elman:
        # Elman (hidden->hidden)
        n.addRecurrentConnection(FullConnection(n["h"], n["h"]))
    else:
        # Jordan (out->hidden)
        n.addRecurrentConnection(FullConnection(n["o"], n["h"]))

    n.sortModules()
    n.reset()

    return n





"""
main
"""
INPUT_LEN = 60*24
OUTPUT_LEN = 5
is_elman = False

TRAINDATA_DIV = 5
parameters = {}

# build rnn
rnn_net = construct_network(INPUT_LEN + 1, OUTPUT_LEN, INPUT_LEN * 2, is_elman)

training_ds = []

rates_fd = open('./hoge.csv', 'r')
exchange_rates = []
for line in rates_fd:
    splited = line.split(",")
    if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>"and splited[0] != "204/04/26" and splited[0] != "20004/04/26":
        time = splited[0].replace("/", "-") + " " + splited[1]
        val = float(splited[2])
        exchange_rates.append([time, val])

data_len = len(exchange_rates)
train_len = len(exchange_rates)/TRAINDATA_DIV

print "data size: " + str(data_len)
print "train len: " + str(train_len)


# hoge = pd.Series(exchange_rates[:][1])
# print exchange_rates[1][:]
# d = {'USDJPY': hoge}
# df = pd.DataFrame(d)
# df.plot(grid=True, figsize=(10,4), y="USDJPY")

x_arr = np.array(xrange(OUTPUT_LEN + 1))
ds  = SupervisedDataSet(INPUT_LEN + 1, OUTPUT_LEN)
for window_s in xrange(train_len-(INPUT_LEN+OUTPUT_LEN)):
    input_rates = []
    output_rates = []
    for i in xrange(window_s, window_s + INPUT_LEN + 1):
        input_rates.append(exchange_rates[i][1])
    for i in xrange(window_s + INPUT_LEN, window_s + INPUT_LEN + OUTPUT_LEN + 1):    
        output_rates.append(exchange_rates[i][1])

    y_arr = np.array(output_rates)
    angle = np.polyfit(x_arr, y_arr, 1)[0]
    ds.addSample(input_rates, [angle])

trainer = BackpropTrainer(rnn_net, **parameters)
trainer.setData(ds)
trainer.train()

del ds  # release memory

# predict
rnn_net.reset()

# frslt = open('../test/rnn_result8.csv', 'w')
# frslt.write(enroll_id_str + "," + str(result[0]) + "\n")        

portfolio = 1000000
portfolio_arr = [[exchange_rates[train_len + INPUT_LEN + OUTPUT_LEN][0], portfolio]]
LONG = 1
SHORT = 2
NOT_HAVE = 3
pos_kind = NOT_HAVE
positions = 0


for window_s in xrange((data_len - train_len) - (INPUT_LEN + OUTPUT_LEN)):
    current_spot = train_len + window_s + INPUT_LEN + OUTPUT_LEN
    input_vals = []
    for i in xrange(train_len + window_s, train_len + window_s + INPUT_LEN + 1):
        input_vals.append(exchange_rates[i][1])
        
    predicted_angle = rnn_net.activate(input_vals)[0]

    print "state " + str(pos_kind)
    print "predicted_angle " + str(predicted_angle)
    if pos_kind == NOT_HAVE:
        if predicted_angle > 0.2:
            pos_kind = LONG
            positions = portfolio / exchange_rates[current_spot][1]
        elif predicted_angle < -0.2:
            pos_kind = SHORT
            positions = portfolio / exchange_rates[current_spot][1]
            sell_val = exchange_rates[current_spot][1]
    else:
        if pos_kind == LONG and predicted_angle < -0.1:
            pos_kind = NOT_HAVE
            portfolio = positions * exchange_rates[current_spot][1]
        elif pos_kind == SHORT and predicted_angle > 0.1:
            pos_kind = NOT_HAVE
            portfolio += positions * sell_val - positions * exchange_rates[current_spot][1]

#    portfolio_arr.append([exchange_rates[current_spot][0], portfolio])
    print str(exchange_rates[current_spot][0]) + " " + str(portfolio)
    
# ig, ax1 = plt.subplots(figsize=(10,4))
# plt.plot(exchange_rates[:][0], exchange_rates[:][1], 'b', label='USDJPY_5M')
# plt.plot(df_mkt.index, df_mkt['actual'], 'g', label='actual')

# plt.grid(True)
# plt.legend(loc=4)

# ax2 = ax1.twinx()
# plt.plot(df_mkt.index, df_mkt['sp500'], 'r', label='s&p500')
# plt.legend(loc=0)
    

    

    

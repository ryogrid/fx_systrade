#!/usr/bin/python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import k_means_

# http://emoson.hateblo.jp/entry/2014/11/03/023224
def dtw(vec1, vec2=None, Y_norm_squared=None, squared=False):
    d = np.zeros([len(vec1)+1, len(vec2)+1])
    d[:] = np.inf
    d[0, 0] = 0
    for i in range(1, d.shape[0]):
        for j in range(1, d.shape[1]):
            cost = abs(vec1[i-1]-vec2[j-1])
            # print(cost + min(d[i-1, j], d[i, j-1], d[i-1, j-1]))
            # print(cost)
            # print(min(d[i-1, j], d[i, j-1], d[i-1, j-1]))
            d[i, j] = cost + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
            
    return [d[-1][-1]]

"""
main
"""
SERIES_LEN = 50
DATA_NUM = 100
CLUSTER_NUM = 10

rates_fd = open('./hoge.csv', 'r')
exchange_rates = []
for line in rates_fd:
    splited = line.split(",")
    if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>"and splited[0] != "204/04/26" and splited[0] != "20004/04/26":
        time = splited[0].replace("/", "-") + " " + splited[1]
        val = float(splited[2])
        exchange_rates.append(val)

tmp_arr = []        
for window_s in xrange(1, DATA_NUM):
    current_spot = DATA_NUM * window_s
    input_data = tmp_arr.append(exchange_rates[current_spot:current_spot + DATA_NUM])

input_data = np.array(tmp_arr)

# overide distance function
k_means_.euclidean_distances = dtw
kmeans_model = KMeans(n_clusters=10, random_state=10).fit(input_data)

labels = kmeans_model.labels_

for label, member in zip(labels, input_data):
        print(label, member)

#!/usr/bin/python
import math
import numpy as np
import random

# http://alexminnaar.com/time-series-classification-and-clustering-with-python.html

def LB_Keogh(s1,s2,r):
    LB_sum=0

    for ind,i in enumerate(s1):
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
            
    return math.sqrt(LB_sum)

def DTWDistance(s1, s2, w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return math.sqrt(DTW[len(s1)-1, len(s2)-1])

def k_means_clust(data,num_clust,num_iter,w=5,org_vals=None):
    centroids=random.sample(data,num_clust)
    counter=0
    for n in range(num_iter):
        counter+=1
        print counter
        assignments={}
        #assign data points to clusters
        for ind,i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                if LB_Keogh(i,j,5)<min_dist:
                    cur_dist=DTWDistance(i,j,w)
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
                        if closest_clust in assignments:
                            assignments[closest_clust].append(ind)
                        else:
                            assignments[closest_clust]=[]
                            #recalculate centroids of clusters

        for key in assignments:
            clust_sum=0
            flag = False
            for k in assignments[key]:
                clust_sum=clust_sum+data[k]
                flag = True
            if flag == True:
                centroids[key]=[m/len(assignments[key]) for m in clust_sum]

    #print
    for key in assignments:
        print("------------------------------")
        for k in assignments[key]:
            for price in org_vals[int(data[k][0]*1000000)]:
                print str(price) + "," ,
            print("")
    
    return centroids

"""
main
"""
SERIES_LEN = 25
DATA_NUM = 250
CLUSTER_NUM = 50
MA_PERIOD = 5
ITR_NUM = 200

rates_fd = open('./hoge.csv', 'r')
exchange_rates = []
for line in rates_fd:
    splited = line.split(",")
    if splited[2] != "High" and splited[0] != "<DTYYYYMMDD>"and splited[0] != "204/04/26" and splited[0] != "20004/04/26":
        time = splited[0].replace("/", "-") + " " + splited[1]
        val = float(splited[2])
        exchange_rates.append(val-110)

ma_vals = []        
for idx in xrange(MA_PERIOD, len(exchange_rates)):
    ma_vals.append(sum(exchange_rates[idx:idx+MA_PERIOD])/MA_PERIOD)

trend_vals = []    
for idx in xrange(2, len(ma_vals)):
    trend_vals.append(ma_vals[idx]-ma_vals[idx-1])
    
tmp_arr = []
to_out_arr = []
i = 0
for window_s in xrange(1, DATA_NUM):
    current_spot = SERIES_LEN * window_s
    tmp_arr.append([0.000001*i] + trend_vals[current_spot:current_spot + SERIES_LEN])
    to_out_arr.append(ma_vals[current_spot:current_spot + SERIES_LEN])
    i+=1

    
input_data = np.array(tmp_arr)

import matplotlib.pylab as plt

centroids=k_means_clust(input_data, CLUSTER_NUM, ITR_NUM, 4, to_out_arr)
for i in centroids:
    plt.plot(i)

plt.show()

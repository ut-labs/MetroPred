#coding=utf8

import os
import datetime
import time
import math
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error

train_path = './dataset/h_train'
flist = os.listdir(train_path)

excepts = [1, 5, 6, 12, 13, 19, 20, 25]
#excepts = []

flist = ['{}.npy'.format(i+1) for i in range(25) if i+1 not in excepts]

data_lists = []
for index, fname in enumerate(flist):
    fpath = os.path.join(train_path, fname)
    data = np.load(fpath)
    data_lists.append(data)

counts = len(data_lists)
print(counts)
res = np.zeros((144, 81, 2))
for i in data_lists:
    res += i

res /= counts
print(res.shape)

def time2str(id, date):
    dt = datetime.datetime.strptime(date, "%Y-%m-%d")
    t1 = time.mktime(dt.timetuple()) + int(id) * 10 * 60
    t2 = t1 + 10 * 60
    t1_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))
    t2_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t2))

    return t1_str, t2_str

with open('test.csv', 'w') as f:
    title = 'stationID,startTime,endTime,inNums,outNums'
    print(title, file=f)
    date = '2019-01-29'
    x, y, z = res.shape
    print(res[0][0])
    for j in range(y):
        for i in range(x):
            t1, t2 = time2str(i, date)
            out_num, in_num = res[i][j] #0出，1进
            out_num = "%.1f"%(out_num)
            in_num = "%.1f"%(in_num)
            print(j, t1, t2, in_num, out_num, sep=',', file=f)


test = np.load('./dataset/h_train/25.npy')
res = res.reshape(1, -1)

test = test.reshape(1, -1)
loss = mean_absolute_error(res, test)
print(loss)

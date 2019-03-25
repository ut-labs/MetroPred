import os
import numpy as np

noises = [1, 5,6, 12,13, 19,20 ]

avg = np.zeros(shape=(144,81,2))
for i in range(1, 26):
    if i in noises: continue
    data = np.load('train/%d.npy'%(i))
    avg += data
avg /= 18

fin = open('test/testA_submit_2019-01-29.csv', 'r')
fout = open('test/testA_submit_2019-01-29_0322.csv', 'w')

first = True
st = 0
time = 0
for line in fin:
    if first:
        first = False
        fout.write(line)
        continue
    line = line[:-1] + ',%.1f,%.1f\n'%(avg[time, st, 0], avg[time, st, 1])
    fout.write(line)
    time += 1
    if time == 144:
        st += 1
        time = 0

fin.close()
fout.close()
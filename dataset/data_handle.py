#coding=utf8

import os
import time
import numpy as np
import datetime
import pandas as pd


def timestr2group(time_str):
    d = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    t = d.timetuple()
    ts = t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec
    return ts

def main():
    dirname = './dataset/Metro_train'
    f_lists = sorted(os.listdir(dirname))
    for index, fname in enumerate(f_lists):
        print(fname)
        #
        fpath = os.path.join(dirname, fname)
        df = pd.read_csv(fpath)

        df['ts'] = df['time'].apply(lambda x: timestr2group(x) // (10 * 60))
        tt = df.groupby(['ts', 'stationID', 'status']).size()
        data = np.zeros((144, 81, 2))

        for (k1, k2, k3), group in tt.iteritems():
            data[k1][k2][k3] = group
        np.save('./dataset/h_train/{}.npy'.format(index + 1), data)


def test_data_extract():
    fpath = './dataset/test/testA_record_2019-01-28.csv'
    df = pd.read_csv(fpath)

    df['ts'] = df['time'].apply(lambda x: timestr2group(x) // (10 * 60))
    tt = df.groupby(['ts', 'stationID', 'status']).size()
    data = np.zeros((144, 81, 2))

    for (k1, k2, k3), group in tt.iteritems():
        data[k1][k2][k3] = group
    np.save('./dataset/h_train/28.npy', data)

if __name__ == '__main__':
    # main()
    test_data_extract()
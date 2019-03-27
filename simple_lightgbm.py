#coding=utf8


import numpy as np


from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb


dataset_path = './dataset/h_train'

val = [24, 25]

datas = []

noises = [1, 5, 6, 12, 13, 19, 20]

for i in range(25):
    fpath = os.path.join(dataset_path, '{}.npy'.format(i + 1))
    data = np.load(fpath)
    #  144 * 81 * 2
    # 换个思路，

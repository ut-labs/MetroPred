# Author: wonder
import os
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.preprocessing import Imputer

from sklearn.metrics import mean_absolute_error


def loadDataset():
    dataset_path = './dataset/train'
    noises = [1, 5, 6, 12, 13, 19, 20]
    datas = []
    for i in range(25):
        if i + 1 in noises: continue
        fpath = os.path.join(dataset_path, '{}.npy'.format(i + 1))
        data = np.load(fpath) # shape: 144 * 81 * 2
        data = data.swapaxes(0,1)
        #data = data.reshape((81*144,2))
        #print("sssss", data.shape)
        datas.append(data)
    #datas = np.concatenate(datas, axis=0)
    #print(datas.shape)
    return np.array(datas)


def loadTrainData(data):
    data_num = len(data)
    #print(data_num)
    #print(data.shape)
    X = []
    Y = []
    for row in range(0, data_num-1):
        X.append(data[row, :, :, :])
        Y.append(data[row+1, :, :, :])
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape((-1, 144, 2))
    Y = Y.reshape((-1, 144, 2))
    X = X.reshape(-1, 288)
    Y = Y.reshape(-1, 288)
    return X, Y

def loadTestData():
    dataset_path = './dataset/h_train'
    fpath = os.path.join(dataset_path, '28.npy')
    data = np.load(fpath) 
    data = data.swapaxes(0,1)

    data[:,:,0], data[:,:,1] = data[:,:,1], data[:,:,0]
    return data
    
def train(X, Y):
    # XGBoost训练过程
    models = []
    for i in range(288):
        model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
        models.append(model)
    
    for t in range(144):
        in_id = 2 * t
        out_id = 2 * t + 1
        models[in_id].fit(X, Y[:, in_id])
        models[out_id].fit(X, Y[:, out_id])
    # 对测试集进行预测
    return models

    #pred = model.predict(val)
    
    # 评估预测结果  
    #print("MAE: %.2f%%" % (mean_absolute_error(Y_val,Y_pred)))
def test(models, X):
    X = X.reshape(81, 288)
    Y = np.zeros((81,144,2))
    for t in range(144):
        in_id = 2 * t
        out_id = 2 * t + 1
        in_pred = models[in_id].predict(X)
        out_pred = models[out_id].predict(X)
        Y[:,t,0] = in_pred
        Y[:,t,1] = out_pred
    return Y
        #print("MAE: %.2f%%" % (mean_absolute_error(Y_val,Y_pred)))
    
if __name__ == '__main__':
    #data = loadDataset()
    #X, Y = loadTrainData(data)
    X_val = loadTestData()
    #models = train(X, Y)

    import pickle
    #with open('xgb_model.pkl', 'wb') as f:
    #    pickle.dump(models, f)
    with open('xgb_model.pkl','rb') as f:
        models = pickle.load(f)
    res = test(models,X_val)


    import time, datetime
    def time2str(id, date):
        dt = datetime.datetime.strptime(date, "%Y-%m-%d")
        t1 = time.mktime(dt.timetuple()) + int(id) * 10 * 60
        t2 = t1 + 10 * 60
        t1_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))
        t2_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t2))

        return t1_str, t2_str

    date = '2019-01-29'
    with open('./results/xgb-ly/29.csv', 'w') as f:
        title = 'stationID,startTime,endTime,inNums,outNums'
        print(title, file=f)
        x, y, z = res.shape
        print(res[0][0])
        for j in range(y):
            for i in range(x):
                t1, t2 = time2str(i, date)
                in_num, out_num = res[i][j]  
                print(j, t1, t2, in_num, out_num, sep=',', file=f)

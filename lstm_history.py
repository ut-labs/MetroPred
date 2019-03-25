# Author: h12345jack
import os
import time
import datetime
import math

import numpy as np
import torch
import torch.nn as nn

import torch.utils.data as Data
import torch.nn.functional as F


from tensorboardX import SummaryWriter

from sklearn.metrics import mean_absolute_error


cuda_device = torch.device('cuda:0')

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
epochs = 801
batch_size = 128
torch.manual_seed(13)

os.system('rm -rf lstm_boardx/*')
writer = SummaryWriter(log_dir='lstm_boardx')


#
class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        bidirectional = True
        dim = 128
        linear_dim = dim * 2 if bidirectional else dim
        self.rnn = nn.LSTM(2, dim,
                           num_layers = 3,
                           batch_first=True,
                           bidirectional=bidirectional,
                           dropout=0.3
                  )
        self.none_linear_layer = nn.Sequential(
            nn.Linear(linear_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.ReLU()
        )

    def forward(self, data):
        result, _ = self.rnn(data)
        result = self.none_linear_layer(result)
        return result

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.weights = torch.nn.Parameter(torch.FloatTensor(144,2))
        self.bias = torch.nn.Parameter(torch.FloatTensor(144,2))
        self.bi_lstm = BiLSTM()

        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data, avg):
        result = self.bi_lstm(data[:,:,:2]) # shape: bs*144*2
        sid = data[:,0,2].long()
        result = result * (1-self.weights) + avg[sid] * self.weights + self.bias
        return result

def main():
    dataset_path = './dataset/h_train'

    val = [24, 25]

    datas = []

    noises = [1, 5, 6, 12, 13, 19, 20]

    avg = np.zeros((144, 81, 2))
    for i in range(1, 26): # 1..24 ? 1..25 ??? 1..25!
    #for i in [4, 11, 18]:
        if i in noises: continue
        fpath = os.path.join(dataset_path, '{}.npy'.format(i))
        data = np.load(fpath)
        avg += data
    avg /= 17
    avg = avg.swapaxes(0,1)
    avg = torch.from_numpy(avg).float().cuda(cuda_device)

    for i in range(25):
        fpath = os.path.join(dataset_path, '{}.npy'.format(i + 1))
        data = np.load(fpath)
        data = torch.from_numpy(data)
        data = torch.einsum("ijk->jik", data)

        ids = torch.from_numpy(np.array([np.ones(144)*i for i in range(81)])).double()
        ids = ids.view(81, 144, 1)

        data = torch.cat([data, ids], dim=2)

        datas.append(data)

    X = []
    for i in range(25 - 1):
        if i + 1 in noises: continue
        if i + 2 in noises: continue
        a = datas[i]
        b = datas[i + 1]
        c = torch.cat((a, b), dim=2)
        X.append(c)

    all_data = torch.cat(X, dim=0).float().cuda(cuda_device)

    a = datas[val[0] - 1]
    b = datas[val[1] - 1]
    val_data = torch.cat((a, b), dim=2).float().cuda(cuda_device)


    torch_dataset = Data.TensorDataset(all_data[:,:,:3], all_data[:,:,3:5])
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers= 0
    )
    #model = BiLSTM()
    model = Model()
    model.cuda(cuda_device)

    crition = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(epochs):
        total_loss = []
        time1 = time.time()

        model.train()

        for step, (batch_x, batch_y) in enumerate(loader):
            optimizer.zero_grad()

            X = batch_x
            y = batch_y
            pred_y = model(X, avg)
            loss = crition(pred_y, y)
            total_loss.append(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()

        model.eval()
        val_X = val_data[:, :, :3]
        val_y = val_data[:, :, 3:5]
        # print(val_y.shape)
        pred_y = model(val_X, avg)
        #pred_y = F.relu(pred_y)
        #print(pred_y.shape)
        val_loss = crition(pred_y, val_y)
        a = pred_y.data.cpu().numpy().reshape(1, -1)
        b = val_y.data.cpu().numpy().reshape(1, -1)

        train_loss = np.mean(total_loss)
        val_loss = val_loss.data.cpu().numpy().mean()
        print("Epoch", epoch,'train loss:', train_loss)
        print("Epoch", epoch,'validation loss:', val_loss )
        print(time.time() - time1)

        writer.add_scalars("scalar/loss", {'train_loss': train_loss}, epoch)
        writer.add_scalars("scalar/loss", {'val_loss': val_loss}, epoch)
        writer.add_scalars("scalar/loss", {'13.5': 13.5}, epoch)
        writer.add_scalars("scalar/loss", {'13.3': 13.3}, epoch)
        writer.add_scalars("scalar/loss", {'13.1': 13.1}, epoch)

        if epoch % 100 == 0:
            fpath = os.path.join(dataset_path, '28.npy')
            test_data = np.load(fpath)
            test_data = torch.from_numpy(test_data)
            test_data = torch.einsum("ijk->jik", test_data)
            ids = torch.from_numpy(np.array([np.ones(144)*i for i in range(81)])).double()
            ids = ids.view(81, 144, 1)

            test_data = torch.cat([test_data, ids], dim=2).float().cuda(cuda_device)

            res = model(test_data, avg)
            res = F.relu(res)
            res = torch.einsum('ijk->jik', res)
            res = res.data.cpu().numpy()

            def time2str(id, date):
                dt = datetime.datetime.strptime(date, "%Y-%m-%d")
                t1 = time.mktime(dt.timetuple()) + int(id) * 10 * 60
                t2 = t1 + 10 * 60
                t1_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))
                t2_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t2))

                return t1_str, t2_str

            date = '2019-01-29'
            with open('./lstm_layers_dropout/{}-{}.csv'.format(date, epoch), 'w') as f:
                title = 'stationID,startTime,endTime,inNums,outNums'
                print(title, file=f)
                x, y, z = res.shape
                print(res[0][0])
                for j in range(y):
                    for i in range(x):
                        t1, t2 = time2str(i, date)
                        out_num, in_num = res[i][j]
                        in_num = '%.3f'%(in_num)
                        out_num = '%.3f'%(out_num)  
                        print(j, t1, t2, in_num, out_num, sep=',', file=f)


if __name__ == '__main__':
    main()

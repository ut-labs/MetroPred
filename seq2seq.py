# Author: h12345jack
import os
import time
import datetime


import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import L1Loss

from sklearn.metrics import mean_absolute_error

import click

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.000
epochs = 5000
batch_size = 128
torch.manual_seed(2333)
#
class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(2, hidden_size, batch_first=True, num_layers=1)

    def forward(self, x):
        x, hid = self.rnn(x)
        return x, hid

class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=1)
        self.embed = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Dropout(0.2)
        )
        self.in_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size)
        )
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_size, 2),
            nn.ReLU()
        )
        
    def forward(self, x, h, c):
        emb = self.embed(x)
        emb = torch.cat((emb, c), dim=2)

        in_x = self.in_layer(emb)

        out_x, hid = self.rnn(in_x, h)
        out_x = self.out_layer(out_x)

        return out_x, hid
                

class Seq2Seq(nn.Module):
    def __init__(self, hidden_size=64):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        bs = x.shape[0]

        context, hidden_state = self.encoder(x)
        
        ret = torch.zeros(x.shape[0], 144, 2).cuda()
        y = torch.zeros(x.shape[0], 1, 2).cuda()

        for di in range(144):
            y, hidden_state = self.decoder(y, hidden_state, context[:,di,:].view(bs, 1, self.hidden_size))
            ret[:,di,:] = y[:,0,:]
        return ret
            
        
@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.option('--valid', '-l', is_flag=True)
@click.option('--name', '-n', default='0323')
def main(valid, name):

    dataset_path = './dataset/h_train'

    val = [24, 25]

    datas = []

    noises = [1, 5, 6, 12, 13, 19, 20]

    for i in range(25):
        fpath = os.path.join(dataset_path, '{}.npy'.format(i+1))
        data = np.load(fpath)
        data = torch.from_numpy(data)
        data = torch.einsum("ijk->jik", data)
        datas.append(data)

    X = []
    for i in range(23 if valid else 24):
        if i+1 in noises: continue
        if i+2 in noises: continue
        a = datas[i]
        b = datas[i+1]
        c = torch.cat((a, b), dim=2)
        X.append(c)

    all_data = torch.cat(X, dim=0).float().cuda()

    a = datas[val[0] - 1]
    b = datas[val[1] - 1]
    val_data = torch.cat((a, b), dim=2).float().cuda()



    model = Seq2Seq()
    model.cuda()

    crition = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(epochs):
        total_loss = []
        time1 = time.time()
        model.train()
        ids = np.arange(all_data.shape[0])
        np.random.shuffle(ids)
        for i in range(all_data.shape[0] // batch_size):
            optimizer.zero_grad()

            ids_batch = ids[i*batch_size : (i+1) * batch_size]
            data_batch = all_data[ids_batch]
            X = data_batch[:,:, :2]
            y = data_batch[:,:, 2:]

            pred_y = model(X)
            loss = crition(pred_y, y)
            total_loss.append(loss.data.cpu().numpy())

            loss.backward()
            optimizer.step()
        
        model.eval()
        val_X = val_data[:, :, :2]
        val_y = val_data[:, :, 2:]
        # print(val_y.shape)
        pred_y = model(val_X)
        # print(pred_y.shape, val_y.shape)
        val_loss = crition(pred_y, val_y)
        a = pred_y.data.cpu().numpy().reshape(1, -1)
        b = val_y.data.cpu().numpy().reshape(1, -1)
        val_loss_sklearn = mean_absolute_error(a, b)
        print(epoch,'train loss:', np.mean(total_loss))
        print(epoch,'val loss:', val_loss.data.cpu().numpy().mean() )
        print(epoch,'sklearn val loss:', val_loss_sklearn)
        print(time.time() - time1)

        if epoch % 10 == 0:
            fpath = os.path.join(dataset_path, '28.npy')
            test_data = np.load(fpath)
            test_data = torch.from_numpy(test_data).float().cuda()
            test_data = torch.einsum("ijk->jik", test_data)

            res = model(test_data)
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
            filepath = 'seq_2_seq_' + name
            if valid: filepath += '_val'
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            with open('./{}/{}-{}.csv'.format(filepath, date, epoch), 'w') as f:
                title = 'stationID,startTime,endTime,inNums,outNums'
                print(title, file=f)
                x, y, z = res.shape
                print(res[0][0])
                for j in range(y):
                    for i in range(x):
                        t1, t2 = time2str(i, date)
                        out_num, in_num = res[i][j]  # 0出，1进
                        print(j, t1, t2, in_num, out_num, sep=',', file=f)


if __name__ == '__main__':
    main()

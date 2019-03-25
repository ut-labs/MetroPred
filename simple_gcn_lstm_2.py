# Author: h12345jack
import os
import time
import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn

import torch.utils.data as Data
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SGConv, ARMAConv, ChebConv
from torch_geometric.data import Data as geoData

from tensorboardX import SummaryWriter


from sklearn.metrics import mean_absolute_error



## 换掉gcn

parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=int, default=1, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')

args = parser.parse_args()


DEBUG = args.debug
print('debug', DEBUG)


cuda_device = torch.device('cuda:3')
log_dir = 'hjj_results_gcn{}'.format(DEBUG)
writer = SummaryWriter(log_dir=log_dir)


os.system("rm -rf {}/*".format(log_dir))

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
epochs = 2000
batch_size = 50
torch.manual_seed(13)
CUDA = True


ALL_DATA = 24 if DEBUG else 25

hidden_dim = 20
#
class BiLSTM(nn.Module):
    def __init__(self, dim=100, dim1=20, dim2=20):
        super(BiLSTM, self).__init__()
        bidirectional = True
        linear_dim = dim * 2 if bidirectional else dim
        self.rnn = nn.LSTM(dim1+dim2, dim,
                           num_layers = 3,
                           batch_first=True,
                           bidirectional=bidirectional,
                           dropout=0.2
                  )

        self.none_linear_layer = nn.Sequential(
            nn.Linear(linear_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.ReLU()
        )

    def forward(self, data):
        result, _ = self.rnn(data)
        result = self.none_linear_layer(result)
        return result

class GCNNet(nn.Module):
    def __init__(self, node_num):
        super(GCNNet, self).__init__()
        dim = hidden_dim
        # 摊平全链接
        # self.node_embedding = nn.Embedding(node_num, dim)
        # self.node_embedding.weight.requires_grad = True
        # self.node_embedding.cuda()
        #
        # self.conv1 = GCNConv(144*2, dim)
        # self.fc = nn.Sequential(
        #     nn.Linear(144 * 2, dim),
        #     nn.ReLU()
        # )


        # 搞个gru
        self.gru_encoder = nn.GRU(2, dim, bidirectional=True, batch_first=True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU()
        )

        # self.conv1 = ChebConv(dim, dim, K=10)
        # self.conv2 = ChebConv(dim, dim, K=10)
        self.conv1 = ARMAConv(dim, dim)
        self.conv2 = ARMAConv(dim, dim)

    def forward(self, graph_data, data):
        x, edge_index, edge_weight = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        # lstm
        # ids = data[:, 0 ,2].view(-1).long()
        # # print(ids.shape)
        # add_x = torch.Tensor(x.cpu().numpy()).cuda()
        # # print(add_x.shape)
        # add_x[ids,:,:] = data[:,:,:2]
        # x = (x + add_x)/2

        edge_index = edge_index.long()
        #
        # x = x.long()
        # x = self.node_embedding(x)
        # x = self.fc(x)

        _, h_n = self.gru_encoder(x)
        h_n = h_n.view(h_n.shape[1], -1)

        x = self.lstm_fc(h_n)
        x = x.view(-1, hidden_dim)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)

        return x


class Model(nn.Module):
    def __init__(self, graph_data):
        super(Model, self).__init__()
        self.lstm = BiLSTM()
        self.gcn = GCNNet(81)
        self.graph_data = graph_data
        self.fc = nn.Sequential(
            nn.Linear(2, hidden_dim)
        )

    def forward(self, data):
        # data bs * 144, 3

        network_emb = self.gcn(self.graph_data, data)
        # print(network_emb.shape)
        # print(network_emb[0], 151)
        # assert  network_emb.shape == (81, hidden_dim)
        x = data[:, :, :2]
        new_x1 = self.fc(x)

        # new_x1 = x

        ids = data[:, :, 2].long()
        ids = ids[:,0]

        new_x2 = network_emb[ids]
        # print(network_embd1(torch.Tensor([0]).long().cuda()), 162) #这里的取之是一样的，loss.backward注释
        # print(new_x2.shape)
        new_x2 = new_x2.view(new_x2.shape[0], 1, new_x2.shape[1])
        new_x2 = new_x2.repeat(1, 144, 1)
        # 50, 144, 20

        # print(new_x2[2, 0, :])
        # print(new_x2[2, 1, :])
        # return
        # print(new_x1.shape, new_x2.shape)

        x = torch.cat([new_x1, new_x2], dim=2)
        res = self.lstm(x)

        return res



def read_graph_csv(fpath='./maps/graph.csv'):
    datas = []
    with open(fpath) as f:
        for line in f.readlines():
            ll = line.split(",")
            datas.append([float(i) for i in ll])
    graph = np.array(datas)
    assert graph.shape == (81, 81)
    return graph


def main():
    dataset_path = './dataset/h_train'

    val = [24, 25]

    datas = []


    noises = [1, 5, 6, 12, 13, 19, 20]
    noises2 = [5, 6, 12, 13, 19, 20]

    for i in range(25):
        fpath = os.path.join(dataset_path, '{}.npy'.format(i + 1))
        data = np.load(fpath)
        #  144 * 81 * 2
        data = torch.from_numpy(data)
        data = torch.einsum("ijk->jik", data)
        # 81 * 144 * 2
        # 浪费内存吧，反正内存不值钱
        ids = torch.from_numpy(np.array([np.ones(144)*i for i in range(81)])).double()
        ids = ids.view(81, 144, 1)

        data = torch.cat([data, ids], dim=2)
        assert data.shape == (81, 144, 3)

        datas.append(data)
    #
    # print(datas[0][0,0,2], datas[0][0,1,2])
    # return?

    avg = torch.zeros(81, 144, 2, dtype=torch.float)
    counts = 0
    # 将均值搞成节点属性
    for i in range(ALL_DATA - 1):
        if i + 1 in noises: continue
        item = datas[i]
        avg += item[:, :, 0: 2].float()
        counts += 1
    avg /= counts

    X = []
    for i in range(24 - 1):
        if i + 1 in noises: continue
        if i + 2 in noises: continue
        a = datas[i]
        b = datas[i + 1]
        c = torch.cat((a, b), dim=2)
        X.append(c)
        # for j in range(2, 8):
        j = 8
        if i + j < 23 and (i+j+1) not in noises:
            a = datas[i]
            b = datas[i + j]
            c = torch.cat((a, b), dim=2)
            X.append(c)

    graph = read_graph_csv()
    graph_floyd = read_graph_csv('./maps/graph_floyd.csv')

    graph_floyd = np.load('./hjj_maps/avg_25.npy')

    rows, cols = graph.nonzero()
    rows = rows.reshape(-1)
    cols = cols.reshape(-1)

    edge_index = torch.tensor([rows, cols], dtype=torch.float).cuda()
    edge_weight = torch.from_numpy(graph_floyd[rows, cols]).float().cuda()


    # x = torch.tensor([i for i in range(81)], dtype=torch.float).cuda()
    #  x就得修改
    x = avg.cuda()

    graph_data = geoData(x, edge_index=edge_index, edge_attr=edge_weight)

    all_data = torch.cat(X, dim=0).float().cuda()

    a = datas[val[0] - 1]
    b = datas[val[1] - 1]
    val_data = torch.cat((a, b), dim=2).float().cuda()


    torch_dataset = Data.TensorDataset(all_data[:,:,:3], all_data[:,:, 3:5])
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers= 0
    )
    model = Model(graph_data)
    model.cuda()

    crition = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(epochs):
        total_loss = []
        time1 = time.time()

        model.train()
        # model.eval()

        for step, (batch_x, batch_y) in enumerate(loader):
            optimizer.zero_grad()

            X = batch_x
            y = batch_y[:, :, :]
            # print(X[0, 1, 2])
            # print(X[0, 2, 2])
            # return 0
            pred_y = model(X)
            loss = crition(pred_y, y)
            total_loss.append(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()

        model.eval()
        val_X = val_data[:, :, :3]
        val_y = val_data[:, :, 3:5]
        # print(val_y.shape)
        pred_y = model(val_X)
        # print(pred_y.shape, val_y.shape)
        val_loss = crition(pred_y, val_y)
        a = pred_y.data.cpu().numpy().reshape(1, -1)
        b = val_y.data.cpu().numpy().reshape(1, -1)
        avg_c = avg.reshape(1, -1)
        val_loss_sklearn = mean_absolute_error(a, b)
        avg_loss = mean_absolute_error(avg_c, b)
        train_loss = np.mean(total_loss)
        val_loss = val_loss.data.cpu().numpy().mean()
        print("Epoch", epoch, 'train loss:', train_loss)
        print("Epoch", epoch, 'validation loss:', val_loss)
        print("Epoch", epoch, 'avg loss:', np.mean(avg_loss))
        print(time.time() - time1)
        writer.add_scalars("scalar/loss", {'train_loss': train_loss}, epoch)
        writer.add_scalars("scalar/loss", {'val_loss': val_loss}, epoch)
        writer.add_scalars("scalar/loss", {'avg_loss': avg_loss}, epoch)
        writer.add_scalars("scalar/loss", {'13.5': 13.5}, epoch)
        writer.add_scalars("scalar/loss", {'13.3': 13.3}, epoch)
        writer.add_scalars("scalar/loss", {'13.1': 13.1}, epoch)

        if epoch % 10 == 0 and not DEBUG:
            fpath = os.path.join(dataset_path, '28.npy')
            test_data = np.load(fpath)

            # data = torch.from_numpy(data)
            # data = torch.einsum("ijk->jik", data)
            # # 81 * 144 * 2
            # # 浪费内存吧，反正内存不值钱
            # ids = torch.from_numpy(np.array([i for i in range(81)])).double()
            # ids = ids.repeat(144, 1)
            # ids = ids.view(81, 144, 1)
            # data = torch.cat([data, ids], dim=2)


            test_data = torch.from_numpy(test_data)
            test_data = torch.einsum("ijk->jik", test_data)
            ids = torch.from_numpy(np.array([i for i in range(81)])).double()
            ids = ids.repeat(144, 1)
            ids = ids.view(81, 144, 1)
            test_data = torch.cat([test_data, ids], dim=2).float().cuda()

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
            with open('./{}/{}-{}.csv'.format(log_dir, date, epoch), 'w') as f:
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
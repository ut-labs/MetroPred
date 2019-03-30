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
import scipy.sparse as sp

cuda_device = torch.device('cuda:3')

CONTAIN_25 = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
epochs = 8001
batch_size = 32
torch.manual_seed(13)

DIM = 128

os.system('rm -rf tb_output/fz/*')
writer = SummaryWriter(log_dir='tb_output/fz')

def normalized_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj

def spy_sparse2torch_sparse(data):
    """
    :param data: a scipy sparse csr matrix
    :return: a sparse torch tensor
    """
    samples=data.shape[0]
    features=data.shape[1]
    values=data.data
    coo_data=data.tocoo()
    indices=torch.LongTensor([coo_data.row,coo_data.col])
    t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
    return t

def calculate_laplacian(adj, lambda_max=1):  
    adj = normalized_adj(adj + sp.eye(adj.shape[0]))
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return spy_sparse2torch_sparse(adj).cuda(cuda_device)

class GC(nn.Module):
    def __init__(self, in_size, nodes, units, adj, output_size):
        super(GC, self).__init__()
        self.in_size = in_size
        self._nodes = nodes
        self._units = units
        self._adj = adj
        self.output_size = output_size
        self.weights = torch.nn.Parameter(torch.FloatTensor(self._units + self.in_size, self.output_size))
        self.bias = torch.nn.Parameter(torch.FloatTensor(self.output_size))

        stdv = 1. / math.sqrt(self.weights.size(1))
        #torch.nn.init.kaiming_normal_(self.weights.data)
        #torch.nn.init.kaiming_normal_(self.bias.data)
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, state):
        inputs = inputs.view(-1, self._nodes, self.in_size)
        state = state.view(-1, self._nodes, self._units)

        x_s = torch.cat((inputs, state), dim = 2)
        input_size = x_s.shape[2]

        x0 = torch.einsum("ijk->jki", x_s)
        x0 = x0.contiguous().view(self._nodes, -1)

        for m in self._adj:
            x1 = torch.spmm(m, x0)
        x = x1.view(self._nodes, input_size, -1)
        x = torch.einsum("ijk->kij", x)
        x = x.contiguous().view(-1, input_size)

        x = torch.mm(x, self.weights)
        x = x + self.bias
        x = x.view(-1, self._nodes, self.output_size)
        x = x.view(-1, self._nodes * self.output_size)
        return x

class TgcnCell(nn.Module):
    def __init__(self, in_size, num_units, adj, num_nodes):
        super(TgcnCell, self).__init__()

        self.in_size = in_size
        self._nodes = num_nodes
        self._units = num_units
        self._adj = []
        self._adj.append(calculate_laplacian(adj))
        self._gc1 = GC(self.in_size, self._nodes, self._units, self._adj, 2 * self._units)
        self._gc2 = GC(self.in_size, self._nodes, self._units, self._adj, self._units)

        self._act = nn.ReLU()


    def forward(self, inputs, state):
        value = torch.sigmoid(self._gc1(inputs, state))
        r, u = torch.split(value, split_size_or_sections=(value.shape[1]//2,value.shape[1]//2), dim=1)
        r_state = r * state
        c = self._act(self._gc2(inputs, r_state))
        new_h = u * state + (1-u) * c
        return new_h, new_h

class WeatherEmbedding(nn.Module):
    def __init__(self, dim):
        super(WeatherEmbedding, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, dim),
        )
    
    def forward(self, x):
        # x.shape = (?, 4)
        return self.fc(x)

class BiLSTM(nn.Module):
    def __init__(self, dim=100, dim_in=81):
        super(BiLSTM, self).__init__()
        bidirectional = True
        linear_dim = dim * 2 if bidirectional else dim
        self.rnn = nn.LSTM(dim_in, dim,
                           num_layers = 2,
                           batch_first=True,
                           bidirectional=bidirectional,
                           dropout=0.2
                  )

        self.none_linear_layer = nn.Sequential(
            nn.Linear(linear_dim, 2 * 81),
            nn.ReLU()
        )

    def forward(self, data):
        result, _ = self.rnn(data)
        result = self.none_linear_layer(result)
        return result

class Model(nn.Module):
    def __init__(self, adj):
        super(Model, self).__init__()

        self.adj = adj
        self.rnn = TgcnCell(2, DIM, adj, 81)
        #self.rnn2 = TgcnCell(DIM, DIM, adj, 81)

        self.weights = torch.nn.Parameter(torch.FloatTensor(DIM, 288))
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        #torch.nn.init.kaiming_normal_(self.weights.data)
        # KaiMing

        self.wthr_embedding = WeatherEmbedding(DIM)
        #self.lstm = BiLSTM(dim = 128, dim_in=81)

        self.avg_w = torch.nn.Parameter(torch.FloatTensor(144, 2))
        stdv = 1. / math.sqrt(self.avg_w.size(1))
        self.avg_w.data = torch.ones(144,2).float().cuda(cuda_device)
        
        #torch.nn.init.kaiming_normal_(self.avg_w)

    def forward(self, inputs, weather, avg, blend):
        #inputs: bs * 144 * 81 * 2
        #weather: bs * 4
        #avg: 144 * 81 * 2

        #wthr_embed = self.wthr_embedding(weather).view(-1, 1, DIM) # bs * dim
        #wthr_embed = wthr_embed.repeat(1, 81, 1)

        states = torch.zeros(inputs.shape[0], 81 * DIM).float().cuda(cuda_device)
        for i in range(144):
            output, states = self.rnn(inputs[:,i,:,:], states)         
        
        last_output = output
        last_output = last_output.view(-1, 81, DIM)
        #print(last_output.shape)
        #print(wthr_embed.shape)
        #last_output = last_output.matmul(self.weights) # -1, 81, 288
        #last_output = torch.cat((last_output, wthr_embed), dim = 2)
        last_output = last_output.matmul(self.weights) # -1, 81, 288
        
        last_output = last_output.view(-1, 81, 144, 2)
        last_output = F.relu(last_output)
        
        if blend:
            result = last_output*(1-self.avg_w) + avg*self.avg_w
        else:
            result = last_output
        
        result = torch.einsum('ijkl->ikjl', result)
        return result

dataset_path = './dataset/h_train'
weathers = np.load('./weather/wthr.npy')

def get_datas():
    
    val = [24, 25]
    datas = []
    noises = [1, 5, 6, 12, 13, 19, 20]

    

    for i in range(25):
        fpath = os.path.join(dataset_path, '{}.npy'.format(i + 1))
        data = np.load(fpath)
        data = torch.from_numpy(data)
        data = data.view(1, 144, 81, 2)
        datas.append(data)
        if i >= 20:
            datas.append(data)
    ###########################################################################
    avg = np.zeros((144, 81, 2))
    tot = 0
    for i in range(21, CONTAIN_25 + 1):
        if i in noises: continue
        fpath = os.path.join(dataset_path, '{}.npy'.format(i))
        data = np.load(fpath)
        avg += data
        tot += 1
    avg /= tot
    avg = torch.from_numpy(avg).float().cuda(cuda_device)
    avg = torch.einsum('ijk->jik', avg)
    

    X = []
    W = []
    for i in range(CONTAIN_25 - 1):
        if i + 1 in noises: continue
        if i + 2 in noises: continue
        a = datas[i]
        for j in [1]:#, 8, 15, 22]:
            if i + j > CONTAIN_25: break
            b = datas[i + j]
            wh = torch.from_numpy(weathers[i+j]).view(1, 4)
            c = torch.cat((a, b), dim=3)
            print('(',i,',',i + j,')')
            X.append(c)
            W.append(wh)
    print('total data: ', len(X))
    all_data = torch.cat(X, dim=0).float().cuda(cuda_device)
    all_wea = torch.cat(W, dim=0).float().cuda(cuda_device)

    a = datas[val[0] - 1]
    b = datas[val[1] - 1]
    val_data = torch.cat((a, b), dim=3).float().cuda(cuda_device)
    val_wea = torch.from_numpy(weathers[val[1] - 1]).view(1, 4).float().cuda(cuda_device)
    return all_data, val_data, all_wea, val_wea, avg


def read_graph_csv(fpath='./maps/graph_floyd.csv'):
    datas = []
    with open(fpath) as f:
        for line in f.readlines():
            ll = line.split(",")
            datas.append([float(i) for i in ll])
    graph = np.array(datas)
    assert graph.shape == (81, 81)
    return graph

def main():

    train_data, val_data, train_wthr, val_wea, avg = get_datas()

    torch_dataset = Data.TensorDataset(train_data[:,:,:,:2], train_data[:,:,:,2:3], train_wthr)
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers= 0
    )
    #adj = np.load('./maps/freqg_all.npy')
    adj = read_graph_csv()
    '''
    for i in range(81):
        if i != 54:
            adj[:, i] = adj[:, i] / np.sum(adj[:, i])
        else:
            adj[:, i] = 0
        adj[i, i] = 0
    '''
    model = Model(adj)
    model.cuda(cuda_device)

    cri_tri = nn.MSELoss()
    cri_val = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(epochs):
        total_loss = []
        time1 = time.time()

        model.train()

        for step, (batch_x, batch_y, batch_w) in enumerate(loader):
            optimizer.zero_grad()

            X = batch_x
            y = batch_y
            pred_y = model(X, batch_w, avg, True)
            loss = cri_tri(pred_y, y)
            total_loss.append(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()

        model.eval()
        val_X = val_data[:,:,:,:2]
        val_y = val_data[:,:,:,2:3]
        
        pred_y = model(val_X, val_wea, avg, True)
        val_loss = cri_val(pred_y, val_y)
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

            test_data = torch.from_numpy(test_data).view(1, 144, 81, 2).float().cuda(cuda_device)
            
            
            wea = torch.from_numpy(weathers[29 - 1]).view(1, 4).float().cuda(cuda_device)
            res = model(test_data, wea, avg, True)
            res = res.view(144, 81, 2)
            res = res.data.cpu().numpy()

            def time2str(id, date):
                dt = datetime.datetime.strptime(date, "%Y-%m-%d")
                t1 = time.mktime(dt.timetuple()) + int(id) * 10 * 60
                t2 = t1 + 10 * 60
                t1_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))
                t2_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t2))

                return t1_str, t2_str

            date = '2019-01-29'
            with open('./results/fz_tgcn/{}-{}.csv'.format(date, epoch), 'w') as f:
                title = 'stationID,startTime,endTime,inNums,outNums'
                print(title, file=f)
                x, y, z = res.shape
                print('------------', res[0][0], '----------------')
                for j in range(y):
                    for i in range(x):
                        t1, t2 = time2str(i, date)
                        out_num, in_num = res[i][j]  # 0出，1进
                        print(j, t1, t2, in_num, out_num, sep=',', file=f)


if __name__ == '__main__':
    main()
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

import random



LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
epochs = 5000
batch_size = 128

dim = 16

torch.manual_seed(13)

#
class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        #self.embedding = nn.Linear(2, dim)
        self.rnn = nn.GRU(2, dim, batch_first=True, num_layers=1)
        self.dim = dim
        #self.fc = nn.Linear(dim, dim)

    def forward(self, data, state): # data.shape: bs * 144 * 2
        #emb = self.embedding(data)
        emb = data
        emb = emb.view(data.shape[0], 1, -1)
        result, state= self.rnn(emb, state)
        return result, state

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, dim).cuda()

class Decoder(nn.Module):
    def __init__(self, dim):
        super(Decoder, self).__init__()
        self.rnn = nn.GRU(dim, dim, batch_first=True, num_layers=1)
        self.out = nn.Linear(dim, 2)
        self.embedding = nn.Linear(2, dim)
        self.dropout = nn.Dropout(0)

        self.attn = nn.Linear(dim * 2, 144)
        self.attn_combine = nn.Linear(dim * 2, dim)

        self.dim = dim

    def forward(self, data, state, encoder_outputs):
        embedded = self.embedding(data)
        embedded = self.dropout(embedded)
        state = torch.einsum("ijk->jik", state)

        attn_weights = self.attn(torch.cat((embedded, state), dim=2))
        attn_applied = torch.bmm(
            attn_weights, encoder_outputs
        )
        state = torch.einsum("ijk->jik", state)
        output = torch.cat((embedded, attn_applied), dim=2)
        output = self.attn_combine(output)
        output = F.relu(output)
        output, state = self.rnn(output, state)
        output = self.dropout(output)
        output = F.relu(self.out(output))
        return output, state, attn_weights
        
teacher_forcing_ratio = 0.5
def train(data, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=144):
    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = 144
    target_length = 144

    encoder_outputs = torch.zeros(batch_size, max_length, dim).cuda()

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            data[:,ei,:], encoder_hidden)
        encoder_outputs[:,ei,:] = encoder_output[:,0,:]

    decoder_input = torch.zeros(batch_size,1,2).cuda()

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target[:,di,:].view(batch_size, 1, 2))
            decoder_input = target[:,di,:].view(batch_size, 1, 2)  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decoder_output
            loss += criterion(decoder_output, target[:,di,:].view(batch_size, 1, 2))

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def model(data, encoder, decoder, val_y = None, crition = None):
    with torch.no_grad():
        encoder_hidden = encoder.initHidden(data.shape[0])
        encoder_outputs = torch.zeros(data.shape[0], 144, dim).cuda()
        for ei in range(144):
            encoder_output, encoder_hidden = encoder(
                data[:,ei,:], encoder_hidden)
            encoder_outputs[:,ei,:] = encoder_output[:,0,:]
        
        decoder_input = torch.zeros(data.shape[0],1,2).cuda()
        decoder_hidden = encoder_hidden
        
        ret = torch.zeros_like(data).cuda()
        for di in range(144):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decoder_output

            ret[:,di,:] = decoder_output[:,0,:]
        if crition != None:
            val_loss = crition(ret, val_y)
        else:
            val_loss = None
    return ret, val_loss





def main():

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
    for i in range(24 - 1):
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


    encoder = Encoder(dim).cuda()
    decoder = Decoder(dim).cuda()
    
    crition = nn.L1Loss()
    en_optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    de_optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(epochs):
        total_loss = []
        time1 = time.time()
        
        ids = np.arange(all_data.shape[0])
        np.random.shuffle(ids)
        for i in range(all_data.shape[0] // batch_size):

            ids_batch = ids[i*batch_size : (i+1) * batch_size]
            data_batch = all_data[ids_batch]
            X = data_batch[:,:, :2]
            Y = data_batch[:,:, 2:]
            loss = train(X, Y, encoder, decoder, en_optimizer, de_optimizer, crition)
            total_loss.append(loss)
        
        
        val_X = val_data[:, :, :2]
        val_y = val_data[:, :, 2:]
        # print(val_y.shape)
        # pred_y = model(val_X)
        # print(pred_y.shape, val_y.shape)
        

        pred_y, val_loss = model(val_X, encoder, decoder, val_y, crition)
        
        
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

            res, _ = model(test_data, encoder, decoder)
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
            data_dir = './seq_2_seq_23_val_16/'
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            with open('./{}/{}-{}.csv'.format(data_dir, date, epoch), 'w') as f:
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

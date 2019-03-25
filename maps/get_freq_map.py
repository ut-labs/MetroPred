import numpy as np
import os

dataset = '../dataset/Metro_train'

def main(idx, filepath):
    g = np.zeros((81,81))
    users_routes_in = {}
    users_routes_out = {}
    users_list = set([])
    f = open(filepath, 'r')
    first = True
    all_data = []
    for line in f:
        if first:
            first = False
            continue
        line = line.strip().split(',')
        
        user_id = line[5]
        status = int(line[4])
        station = int(line[2])
        all_data.append((user_id, status, station))
        users_list.add(user_id)
    f.close()
    print('all_data len: ', len(all_data))
    print('users num: ', len(users_list))

    for user_id, status, station in all_data:
        if status == 1:
            if user_id not in users_routes_in:
                users_routes_in[user_id] = []
            users_routes_in[user_id].append(station)
        else:
            if user_id not in users_routes_out:
                users_routes_out[user_id] = []
            users_routes_out[user_id].append(station)
    
    for user_id in users_list:
        in_tot = len(users_routes_in[user_id]) if user_id in users_routes_in else 0
        out_tot = len(users_routes_out[user_id]) if user_id in users_routes_out else 0
        for i in range(min(in_tot, out_tot)):
            station_a = users_routes_in[user_id][i]
            station_b = users_routes_out[user_id][i]
            g[station_a][station_b] += 1

    np.save('freqg_%d.npy'%(idx), g)

if __name__ == '__main__':
    for i in range(1, 26):
        print(i, 'begin')
        filepath = os.path.join(dataset, 'record_2019-01-%02d.csv'%(i))
        main(i, filepath)

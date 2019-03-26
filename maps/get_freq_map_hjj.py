import os

import numpy as np

from collections import defaultdict

dataset = '../dataset/Metro_train'


def main(idx, filepath):
    g = np.zeros((81, 81))
    users_routes_in = defaultdict(list)
    users_routes_out = defaultdict(list)
    users_list = set()
    all_data = []
    with open(filepath, 'r') as f:
        for index, line in enumerate(f.readlines()):
            if index == 0:
                continue

            line = line.strip().split(',')

            user_id = line[5]
            status = int(line[4])
            station = int(line[2])
            all_data.append((user_id, status, station))
            users_list.add(user_id)

    print('all_data len: ', len(all_data))
    print('users num: ', len(users_list))

    for user_id, status, station in all_data:
        if status == 1:
            users_routes_in[user_id].append(station)
        else:
            users_routes_out[user_id].append(station)

    index = 0
    for user_id in users_list:
        # print(len(users_routes_in[user_id]), 40)
        # print("*" * 10)
        a = users_routes_in[user_id]
        b = users_routes_out[user_id]


        # if len(a) != len(b):
        #     print(a)
        #     print(b)
        #     print('=' * 10)
        #     index += 1

        if len(a) == len(b):
            for i,j in zip(a, b):
                g[i][j] += 1
        # in_tot = len(users_routes_in[user_id]) if user_id in users_routes_in else 0
        # out_tot = len(users_routes_out[user_id]) if user_id in users_routes_out else 0
        # for i in range(min(in_tot, out_tot)):
        #     station_a = users_routes_in[user_id][i]
        #     station_b = users_routes_out[user_id][i]
        #     g[station_a][station_b] += 1
    # print(index)

    np.save('../hjj_maps/freqg_%d.npy' % (idx), g)


if __name__ == '__main__':
    for i in range(1, 26):
        print(i, 'begin')
        filepath = os.path.join(dataset, 'record_2019-01-%02d.csv' % (i))
        main(i, filepath)

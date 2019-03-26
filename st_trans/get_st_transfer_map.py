import numpy as np
import os

dataset = '../dataset/Metro_train'

def get_time_144(time):
    time.split(' ')[1].split(':')
    time = int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
    time = time // 600
    return time

def main(idx, filepath):
    in_g = np.zeros((144, 81))

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
        time = get_time_144(line[0])
        all_data.append((user_id, status, time, station))
        users_list.add(user_id)
    f.close()
    print('all_data len: ', len(all_data))
    print('users num: ', len(users_list))

    for user_id, status, time, station in all_data:
        if status == 1:
            if user_id not in users_routes_in:
                users_routes_in[user_id] = []
            users_routes_in[user_id].append((time, station))
        else:
            if user_id not in users_routes_out:
                users_routes_out[user_id] = []
            users_routes_out[user_id].append(time, station)
    
    for user_id in users_list:
        in_tot = len(users_routes_in[user_id]) if user_id in users_routes_in else 0
        out_tot = len(users_routes_out[user_id]) if user_id in users_routes_out else 0
        if in_tot != out_tot: continue
        
        for i in range(in_tot):
            if users_routes_in[user_id][i][0] > users_routes_out[user_id][i][0]: break
            (time_in, station_in) = users_routes_in[user_id][i]
            (time_out, station_out) = users_routes_out[user_id][i]

            in_g[time_in][station_in] += 1
            out_g[time_out][station_out] += 1
            

        

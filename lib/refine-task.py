#!/usr/bin/python3

# inputs: task_name, resolution, quantity, lower_padding, upper_padding
# generates params.dat file for new task.refine
# and puts it into new directory: task_name.refine

# python3 lib/refine-task.py tasks/triangle0 10 EE_reduced

import sys
import os
import shutil

import pandas as pd
import numpy as np

def find_transition(x, y, yt):

    xmax = max(x)
    xmin = min(x)
    ymax = max(y)
    ymin = min(y)
    found = False
    for i, yi in enumerate(y):
        if i==0:
            continue
        if (y[i] > yt) and (y[i-1] < yt):
            xmax = x[i]
            xmin = x[i-1]
            ymax = y[i]
            ymin = y[i-1]
            found=True
            break # only want to find one transition
    if not found:
        print("Threshold not found")
        return None
    xt = xmin + (xmax-xmin) * (yt-ymin)/(ymax-ymin)
    return xt

def add_task_to_queue(master, path):
    # checks if task is not finished nor ongoing
    # appends task to queue
    # with open(f"{master}/finished-tasks.log", 'r') as file:
    #     finished_lines = file.readlines()
    # with open(f"{master}/ongoing-tasks.log", 'r') as file:
    #     ongoing_lines = file.readlines()
    with open(f"{master}/queued-tasks.log", 'r') as file:
        queued_lines = file.readlines()
    
    add_task = True
    for line in queued_lines:
        if line.__contains__(path):
            add_task = False

    if add_task:
        with open(f"{master}/queued-tasks.log", 'a') as file:
            file.write(f"{path}\n")
        print("Task added")
        return;
    print("Task not added")

def remove_task_from_queue(master, path):
    with open(f"{master}/queued-tasks.log", 'r') as file:
        queued_lines = file.readlines()
    
    new_queue = []
    for line in queued_lines:
        if not line.__contains__(path):
            new_queue.append(line)
    
    with open(f"{master}/queued-tasks.log", 'w') as file:
        file.writelines(new_queue)


def calc_additional_errors(df_row, err):
    if err == 'EE':
        return (df_row['En'] - df_row['E0'])/(df_row['Emed'] - df_row['E0'])
    if err == 'EE_reduced':
        return (df_row['En_reduced'] - df_row['E0_reduced'])/(df_row['Emed_reduced'] - df_row['E0_reduced'])
    return None

def main():

    args = sys.argv
    print("args: ", args)
    task_name = args[1]

    # how many points to do the refinement
    # suggestion: 10
    resolution = int(args[2]) if len(args)>2 else 10
    
    quantity = args[3] if len(args)>3 else 'EE'

    # padding above minimum
    # suggestion: 10
    lower_padding = int(args[4]) if len(args)>4 else 10

    # padding below max 
    # suggestion: 10
    upper_padding = int(args[5]) if len(args)>5 else 10

    df_tot = pd.read_csv(f'{task_name}/data.log', sep='\t')
    df_tot['dt'] = df_tot['tf']/df_tot['M']

    cols = df_tot.columns

    if quantity not in cols:
        df_tot[quantity] = df_tot.apply(lambda x: calc_additional_errors(x, quantity), axis=1)

    custom_args = cols[5]
    # for k, g in df_tot.groupby(by=['method', 'fid', custom_args]):
    #     print(k)
    
    df_refine = pd.DataFrame({})

    for k, g in df_tot.groupby(by=['method', 'fid', custom_args]):
        
        df_temp = g.copy().drop_duplicates()
        gkey = {'method':k[0], 'fid':k[1], custom_args:k[2]} 
        # find_transition(x, y, yt)
        df_arr = g[['dt', quantity]].sort_values('dt')
        # print(df_temp[['method','tf','M','fid', custom_args]])
        # print(df_arr.dt.to_numpy())
        # print(df_arr[quantity].to_numpy())

        low_lim = np.min(df_arr[quantity])*lower_padding
        # up_lim = np.max(df_arr[quantity])/upper_padding
        up_lim = 1.0/upper_padding

        dt0 = find_transition(df_arr.dt.to_numpy(), np.log(df_arr[quantity]).to_numpy(), np.log(low_lim))
        dt1 = find_transition(df_arr.dt.to_numpy(), np.log(df_arr[quantity]).to_numpy(), np.log(up_lim))
        if dt0 is None or dt1 is None:
            print("Failed to find transition")
            print(f"FLAG: {k}")
            exit(-1)
            
        dt_arr = np.linspace(dt0, dt1, resolution)
        df_M = pd.DataFrame({'M':np.int64(np.max(g.tf)/dt_arr)})
        # print(df_temp)
        df_temp.drop('M', axis=1, inplace=True)
        # print(df_temp.columns)
        df_temp = df_temp.assign(tf=np.max(g.tf), **gkey).drop_duplicates().merge(df_M, how='cross')
        # print(df_temp.columns)
        df_refine = pd.concat([df_refine, df_temp], ignore_index=True)
    
    # df_refine['Split']=df_refine['Llocal']
    df_p = df_refine.sample(frac=1, ignore_index=True).drop_duplicates()[['method','tf','M','fid', custom_args]]

    new_task = f'{task_name}-refined'
    os.makedirs(new_task, exist_ok=True)

    df_p.to_csv(f'{new_task}/params.dat', sep='\t')
    shutil.copyfile(f"{task_name}/hargs.py", f"{new_task}/hargs.py")
    
    remove_task_from_queue(".", task_name)
    add_task_to_queue(".", new_task)
    




if __name__ == '__main__':
    main()

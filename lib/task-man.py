# Version 0.3.1
# MPC version, with MultiPartyTrotter and MBL metrics
# Debugged -> now runs

# directory structure
#       each directory has the same version of integrator
#   - master
#       - lib
#           - SA, AT, UT, DT, utils
#           - task-man.py
#           - config.py  # defines CPU_LOAD, etc
#       - tasks
#           - task0
#               - params.dat
#                   - method\t tf\t M\t fid\t[args]  
#               - hargs.py    # compiles the arguments in params.dat
#                                to build hamiltonians
#               - data.log
#               - progress.log 
#           - task1
#           - task2
#           - ...
#       - queued-tasks.log   # add new tasks to this file
#       - ongoing-tasks.log  # tasks being currently exectuted are shown here
#       - finished-tasks.log # completed tasks listed here  


# features 
#   - add task 
#       - pandas : generates params.dat
#       - custom code : generates hargs.py  

import numpy as np
import ast
import importlib
import inspect
from config import QUTIP_CPU_LOAD, TASK_THREADS

from qutip import *
qutip.settings.num_cpus = QUTIP_CPU_LOAD
from qutip.states import ket2dm

from AdiabaticTrotter import AdiabaticTrotter
from UltraTrotter import UltraTrotter
from DistributedTrotter import DistributedTrotter
from MultiPartyTrotter import MultiPartyTrotter
from utils import *
from datetime import datetime
import multiprocessing as mp
import re
import os

def pool_compute_task_mpc(task, method, tf, M, fid, args, header_order, metrics, q):
    # hargs_path = f"tasks.{task}.hargs"
    # hargs_module = importlib.import_module(hargs_path)
    # header_order_norms = header_order.copy()
    def put_results(results, metrics, now):
        datetime_now = now.strftime("%H:%M:%S.%f")[:-3]
        output = datetime_now
        for sh in header_order:
            output += f'\t{results[sh]}'
        for sh in metrics:
            output += f'\t{results[sh]}'
        q.put(output)
    
    hargs_spec = importlib.util.spec_from_file_location("Hargs", f"{task}/hargs.py")
    hargs_mod = importlib.util.module_from_spec(hargs_spec)
    hargs_spec.loader.exec_module(hargs_mod)

    # Hams = make_hamiltonian_from_spec(*hargs_mod.Hargs.hargs_from_file(args))
    ham_norm = hargs_mod.Hargs.extra_args['H_norm']
    ham_norm_list = []
    if type(ham_norm) == type(list()): # list type
        ham_norm_list = ham_norm
    elif type(ham_norm) == type(""): # string type
        ham_norm_list = [ham_norm]
    psi_def =  hargs_mod.Hargs.extra_args['psi0']

    propagate = True # always propagates unless specified
    if 'propagate' in hargs_mod.Hargs.extra_args:
        propagate = hargs_mod.Hargs.extra_args['propagate']
    psi0 = None
    
    
    results_pre = {'method':method, 'tf':tf, 'M':M, 'fid':fid, 'args':args}
    results = {}

    # hnorms = {} # ex: '||HF||_max' : ('||HF||', 'max')
    new_metrics = []
    for norm in ham_norm_list:
        for m in metrics:
            new_m = m
            if m.__contains__('||'): # if its an Hamiltonian norm
                new_m = f'{m}_{norm}'
                # hnorms[new_m] = (m, norm)
            results[new_m] = 'NA' # load new metrics (if norm is renamed) to 'NA'
            new_metrics.append(new_m)
    for key, val in results_pre.items():
        results[key] = val
    
    # try:
    mpc_solver =  MultiPartyTrotter(hargs_mod.Hargs.hargs_from_file(args), method=method)

    if psi_def is None:
        psi0 = None
    elif psi_def == 'mpc':
        psi0 = MultiPartyTrotter.get_mpc_psi0(mpc_solver.sublocals, mpc_solver.system_spec)
    
    if propagate:
        mpc_solver.propagate(tf, M, fid, psi0=psi0)
    else: 
        for rm_metric in ['En', 'OE']:
            # remove metrics that require propagation
            if rm_metric in metrics:
                metrics.remove(rm_metric) 
    
    for norm in ham_norm_list:
        results_mpc = mpc_solver.calc_errors(metrics, norm)
        
        for key, val in results_mpc.items():
            new_key = key
            if key.__contains__('||'): # if its a norm
                new_key = f'{key}_{norm}'
            results[new_key] = val
    put_results(results, new_metrics, datetime.now())
    return 1
    # except:
    #     put_results(results, datetime.now())
    #     return 0


def logger_listener(q, job_tot, path):
    # result is a string
    #
    job_p = 0.0
    start = datetime.now()
    with open(f"{path}/data.log", 'w') as logf, open(f"{path}/progress.log", 'w') as pf:
        while 1:
            m = q.get()
            if m == 'kill':
                # logf.write('\nfinished\n')
                break

            job_p += 1
            now = datetime.now()
            datetime_now = now.strftime("%H:%M:%S.%f")[:-3]
            delta = now-start
            p = (job_p/job_tot)
            eta = delta * (1-p)/p
            # writes log line, then writes progress line
            logf.write(str(m) + '\n')
            logf.flush()
            pf.write(f'\r{datetime_now} { 100*p:.4}% ... ETA {str(eta)}      ')
            
            pf.flush()

def process_task(pool, task):
    jobs = []
    manager = mp.Manager()
    q = manager.Queue()

    with open(f"{task}/params.dat") as file_param:
        lines = file_param.readlines()
        for line in lines:
            m = re.split('\t', line.replace('\n', ''))
            # print(m)
            if len(m) > 4 and (not m[1].__contains__('method')):
                method = m[1]
                tf = float(m[2])
                M = int(m[3])
                fid = float(m[4])
                args = ' '.join(m[5:])

                jobs.append((method, tf, M, fid, args))

    job_tot = len(jobs)+2

    logger = pool.apply_async(logger_listener, (q, job_tot, task))
    
    hargs_path = f"{task}/hargs.py"
    # print(os.getcwd())
    hargs_spec = importlib.util.spec_from_file_location("Hargs", hargs_path)
    hargs_mod = importlib.util.module_from_spec(hargs_spec)
    hargs_spec.loader.exec_module(hargs_mod)
    custom_args = inspect.getfullargspec(hargs_mod.Hargs.h_spec_build).args

    header_order = ['method', 'tf', 'M', 'fid', 'args']
    metrics  = hargs_mod.Hargs.metrics

    header = ''
    ham_norm = hargs_mod.Hargs.extra_args['H_norm']
    ham_norm_list = []
    if type(ham_norm) == type(list()): # list type
        ham_norm_list = ham_norm
    elif type(ham_norm) == type(""): # string type
        ham_norm_list = [ham_norm]
    new_metrics = []
    for norm in ham_norm_list:
        for m in metrics:
            new_m = m
            if m.__contains__('||'): # if its an Hamiltonian norm
                new_m = f'{m}_{norm}'
            new_metrics.append(new_m)
    for sh in [*header_order, *new_metrics]:
        if not sh.__contains__('args'):
            header += f'\t{sh}'
        else:
            header += f'\t{custom_args}'

    q.put(header)

    pool_jobs = []

    for jb in jobs:
        job = pool.apply_async(pool_compute_task_mpc, (task, *jb,header_order, metrics, q ))
        pool_jobs.append(job)
    
    succ = 0
    for job in pool_jobs:
        succ += job.get()
    
    q.put(f"\n\nSuccessful jobs: {succ}")
    
    q.put('kill')
    # pool.close()
    # pool.join()

    return succ


def main():
    pool = mp.Pool(TASK_THREADS+1)
    manager = mp.Manager()
    q = manager.Queue()

    tasks_done = False
    queue_index = 0
    
    while not tasks_done:
        with open("queued-tasks.log", 'r') as file_q:
            # get current task
            lines = file_q.readlines()
            # print(lines)
            if queue_index < len(lines):
                # print(queue_index)
                line = lines[queue_index].replace('\n', '')
            else:
                tasks_done = True
                break
        if tasks_done: break
         
        start = datetime.now()
        dt_start = start.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
        with open("ongoing-tasks.log", 'w') as file_on:
                # tag task as ongoing 
                file_on.write(f"{dt_start}\t{line}\n")
        
        # process task
        process_task(pool, line)

        end = datetime.now()
        dt_end = end.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
        with open("finished-tasks.log", 'a') as file_fin:
                # tag task as finished
                file_fin.write(f"{dt_start}\t{dt_end}\t{end-start}\t{line}\n")
        
        queue_index +=1
    print("Finished all tasks")
    pool.close()
    pool.join()
    
if __name__ == '__main__':
    main()



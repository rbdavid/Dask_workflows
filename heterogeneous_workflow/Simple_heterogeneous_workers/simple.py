#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import sys
import glob
import time

import subprocess
from subprocess import CalledProcessError

import platform
import dask.config
from distributed import Client, Worker, as_completed, get_worker

#######################################
### DASK RELATED FUNCTIONS
#######################################

def get_num_workers(client):
    """ Get the number of active workers
    :param client: active dask client
    :return: the number of workers registered to the scheduler
    """
    scheduler_info = client.scheduler_info()

    return len(scheduler_info['workers'].keys())


def disconnect(client, workers_list):
    """ Shutdown the active workers in workers_list
    :param client: active dask client
    :param workers_list: list of dask workers
    """
    client.retire_workers(workers_list, close_workers=True)
    client.shutdown()


def gpu_submit_pipeline(run_num):
    """
    """
    start_time = time.time()
    worker = get_worker()
    print(start_time,platform.node(),worker.id)
    exec_command = 'python3 /gpfs/alpine/bif135/proj-shared/rbd_work/dask_testing/Min_and_analysis/minimization.py /gpfs/alpine/bif135/world-shared/species/smagellanicum/af_mod/Sphm01G000100.1/model_3_ptm_20211001_725784.pdb "non_hydrogen" /gpfs/alpine/proj-shared/bif135/rbd_work/dask_testing/Min_and_analysis/testing/run_%s'%(run_num)

    try:
        completed_process = subprocess.run(f'{exec_command}',shell=True,capture_output=True,check=True) # ,cwd=working_directory
        print(start_time, time.time(), platform.node(), worker.id, run_num, 0)
        return worker.id, start_time, time.time(), run_num, 0

    except CalledProcessError as e:
        print(start_time, time.time(), platform.node(), worker.id, run_num, 1)
        return worker.id, start_time, time.time(), run_num, 1

    except Exception as e:
        print(start_time, time.time(), platform.node(), worker.id, run_num, 1)
        return worker.id, start_time, time.time(), run_num, 2

def cpu_submit_pipeline(run_num):
    """
    """
    start_time = time.time()
    worker = get_worker()
    print(start_time,platform.node(),worker.id)
    structure_file = glob.glob('/gpfs/alpine/bif135/proj-shared/rbd_work/dask_testing/Min_and_analysis/testing/run_%s/Sphm01G000100.1/*min_00.pdb'%(run_num))[0]
    exec_command = f'python3 /gpfs/alpine/bif135/proj-shared/rbd_work/dask_testing/Min_and_analysis/centered.py {structure_file}'

    try:
        completed_process = subprocess.run(f'{exec_command}',shell=True,capture_output=True,check=True) # ,cwd=working_directory
        print(start_time, time.time(), platform.node(), worker.id, run_num, 0)
        return worker.id, start_time, time.time(), run_num, 0

    except CalledProcessError as e:
        print(start_time, time.time(), platform.node(), worker.id, run_num, 1)
        return worker.id, start_time, time.time(), run_num, 1

    except Exception as e:
        print(start_time, time.time(), platform.node(), worker.id, run_num, 2)
        return worker.id, start_time, time.time(), run_num, 2


#######################################
### MAIN
#######################################

if __name__ == '__main__':
   
    client = Client(scheduler_file=sys.argv[1],timeout=5000,name='all_tsks_client')
    runs = list(range(64))
    
    # pre-processing of first gen
    
    
    task_futures = client.map(gpu_submit_pipeline, runs, pure=False, resources={'GPU':1}) 
    gpu_ac = as_completed(task_futures)
    cpu_futures = []
    for i, finished_task in enumerate(gpu_ac):
        worker_id, start_time, stop_time, run_num, return_code = finished_task.result()
        print(worker_id, start_time, stop_time, run_num, return_code)
        if return_code==0:
            cpu_future = client.submit(cpu_submit_pipeline, run_num, pure=False, resources={'CPU':1})
            cpu_futures.append(cpu_future)

    # pre-processing of second gen
    cpu_ac = as_completed(cpu_futures)
    for i, finished_task in enumerate(cpu_ac):
        worker_id, start_time, stop_time, run_num, return_code = finished_task.result()
        print(worker_id, start_time, stop_time, run_num, return_code)

    #results = client.gather(cpu_futures)
    #for result in results:
    #    print(result[0],result[1],result[2],result[3],result[4])
    
    sys.stdout.flush()  # Because Summit needs nudged
    sys.stderr.flush()
    
    client.shutdown()


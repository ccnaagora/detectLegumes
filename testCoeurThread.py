from concurrent.futures import ProcessPoolExecutor
import os
from multiprocessing import Process
import os

import psutil


def worker(num):
    pid = os.getpid()
    p  = psutil.Process(pid)
    cpu_occupation = psutil.cpu_percent(interval = 0.1 , percpu=True)
    numero = cpu_occupation.index(min(cpu_occupation))
    print(f"pid={pid}Worker {num} s'exécute sur le cœur {numero}")
    print(cpu_occupation[numero])

if __name__ == "__main__":
    num_cores = os.cpu_count()
    print(f"num_cores = {num_cores}")
    processes = []
    for i in range(num_cores):
        p = Process(target=worker, args=[i,])
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
'''
def worker(num):
    print(f"Worker {num} s'exécute sur le cœur {os.sched_getaffinity(0)}")

if __name__ == "__main__":
    print(os.cpu_count())
    i=0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        print(range(os.cpu_count()))
        #executor.map(worker, range(os.cpu_count()))
        executor.map(worker, range(os.cpu_count()))
        i=i+1
'''
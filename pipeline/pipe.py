from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .partition import _split_module
from threading import Thread

# ASSIGNMENT 4.2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    for time in range(num_batches + num_partitions - 1):
        yield [
                (time - part_idx, part_idx) 
                for part_idx in range(
                    max(0, time - num_batches + 1), 
                    min(time + 1, num_partitions)
                )
            ]

result_ = {}
def worker_func(worker_idx, func):
    result_[worker_idx] = func()

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        self.num_pipe_layers = len(self.devices)

    # ASSIGNMENT 4.2
    def forward(self, x):
        # file = open("out.txt", "w")
        actived_split_size = min(x.shape[0], self.split_size) # for self.split_size > batch_size
        mini_batch_size = (x.shape[0] + actived_split_size - 1) // actived_split_size
        batches = [ x[i * mini_batch_size: (i + 1) * mini_batch_size] for i in range(actived_split_size) ]
        for cur_sch in _clock_cycles(actived_split_size, self.num_pipe_layers):
            # print(cur_sch)
            # file.write(str(cur_sch))
            self.compute(batches, cur_sch)
        # file.close()
        return torch.cat(batches, 0)

    # ASSIGNMENT 4.2
    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        partitions = self.partitions
        devices = self.devices
        
        threads = []
        # Step 1: push each work into a worker
        for (batch_idx, worker_idx) in schedule:
            t = Thread(
                    target = worker_func, 
                    args = (worker_idx, lambda: partitions[worker_idx](batches[batch_idx])), 
                    daemon = True,
                )
            t.start()
            threads.append(t)

        # Step 2: Wait for the worker to end
        for t in threads:
            t.join()
        
        # Step 3: fetch the result
        for (batch_idx, worker_idx) in schedule:
            batches[batch_idx] = result_[worker_idx]
            if worker_idx + 1 < self.num_pipe_layers:
                # put it to the next device
                batches[batch_idx] = batches[batch_idx].to(devices[worker_idx + 1])
            else:
                print("Find error")
    
from random import Random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


# ASSIGNMENT 4.1
class Partition():
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        '''Given index, get the data according to the partitioned index'''
        return self.data[index]


# ASSIGNMENT 4.1
class DataPartitioner():
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        
        # Create shuffled indices
        data_len = len(data)
        rng.shuffle(self.data)

        # Partition indices according to sizes
        start = 0
        for size in sizes:
            partition_size = int(size * data_len)
            self.partitions.append(self.data[start : start + partition_size])
            start += partition_size

    def use(self, partition):
        ''' Return a simple dataset class `Partition` by original data and partitioned indices '''
        return Partition(self.partitions[partition], list(range(len(self.partitions[partition]))))


# ASSIGNMENT 4.1
def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """ Partitioning training dataset of the Machine Translation

    Returns:
        DataLoader: partitioned dataloader
    """
    partitioner = DataPartitioner(dataset, [1.0 / world_size] * world_size)
    partition = partitioner.use(rank)

    return DataLoader(partition, batch_size = batch_size // world_size, collate_fn = collate_fn)
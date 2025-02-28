import sys
import os
import pytest
from pathlib import Path

cousin_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(cousin_dir))

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent

from data_parallel.dataset import partition_dataset
from torch import nn
import torch


@pytest.mark.a4_1_1
@pytest.mark.parametrize("total_num", [64, 128, 512])
@pytest.mark.parametrize("split_size", [2, 4, 8])
def test_data_partition(total_num, split_size):
    data = [torch.tensor(i) for i in range(total_num)]
    bsz = total_num // split_size
    partitions = [partition_dataset(i, split_size, data, bsz) for i in range(split_size)]
    visited_sets = set()
    for part in partitions:
        for d in part:
            assert d not in visited_sets
            visited_sets.add(d)


@pytest.mark.a4_1_2
def test_gradient_accumulation():
    weight0 = torch.load(f"{current_dir}/model0_gradients.pth")
    weight1 = torch.load(f"{current_dir}/model1_gradients.pth")
    weight2 = torch.load(f"{current_dir}/model2_gradients.pth")
    weight3 = torch.load(f"{current_dir}/model3_gradients.pth")

    assert len(weight0) == len(weight1)
    assert weight0.keys() == weight1.keys()
    assert len(weight0) == len(weight2)
    assert weight0.keys() == weight2.keys()
    assert len(weight0) == len(weight3)
    assert weight0.keys() == weight3.keys()
    for key in weight0.keys():
        assert torch.sum(weight0[key] != weight1[key]) == 0, f"No sync on gradient {key}, 0 != 1"
        assert torch.sum(weight0[key] != weight2[key]) == 0, f"No sync on gradient {key}, 0 != 2"
        assert torch.sum(weight0[key] != weight3[key]) == 0, f"No sync on gradient {key}, 0 != 3"

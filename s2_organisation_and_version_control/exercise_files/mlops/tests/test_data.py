import os

print(__package__)

from .utils.helper import _PATH_DATA
from .utils.helper import _PROJECT_ROOT


import sys
import pytest
sys.path.append(os.path.join(_PROJECT_ROOT, "src"))
from features.build_features import mnist
# from ..src.features.build_features import mnist ## doesn't work why?

@pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA, 'processed/train.npz'))\
                    or not os.path.exists(os.path.join(_PATH_DATA, 'processed/test.npz')),\
                    reason="data doesn't exist")
def test_data():
    train_dataset, test_dataset = mnist()
    N_train = 40000
    N_test = 5000
    count = [0 for _ in range(10)]
    assert len(train_dataset) == N_train and len(test_dataset) == N_test

    for tr in train_dataset:
        assert tuple(tr[0].shape) == (1,28,28)
        assert tr[1].item() in [0,1,2,3,4,5,6,7,8,9]
        count[tr[1].item()] = 1

    assert sum(count) == 10
    count = [0 for _ in range(10)]
    
    for ts in test_dataset:
        assert tuple(ts[0].shape) == (1,28,28)
        assert ts[1].item() in [0,1,2,3,4,5,6,7,8,9]
        count[ts[1].item()] = 1
    
    assert sum(count) == 10




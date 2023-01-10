import os
import pytest
from .utils.helper import _PATH_DATA
from .utils.helper import _PROJECT_ROOT
import sys
sys.path.append(os.path.join(_PROJECT_ROOT, "src"))
from models.model import MyAwesomeModel
import torch

def test_model():
    model = MyAwesomeModel()
    with torch.no_grad():
        inputs = torch.randn(1,1,28,28)
        output = model(inputs)
        assert tuple(output.shape) == (1,10)

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match=r"Expected 4D tensor, got [1-9][0-9]*D tensor instead"):
        model(torch.randn(1,28,28))

